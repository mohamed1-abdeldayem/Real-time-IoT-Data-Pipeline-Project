#!/usr/bin/env python3
# [translate: قراءة بيانات محطات الطاقة الشمسية من Kafka ومعالجتها وتحليلها]

import json
from kafka import KafkaConsumer
import pandas as pd
from datetime import datetime, timezone
import pytz

KAFKA_BOOTSTRAP_SERVERS = ['kafka-broker-1:9092', 'kafka-broker-2:9093']
KAFKA_TOPIC = 'solar-stations'           # مغيرين الاسم ليخص محطات الشمس
CONSUMER_GROUP = 'solar-farm-consumer-group'   # اسم جروب جديد

def validation_and_cleaning(df):
    required_cols = ["station_id", "timestamp", "solar_irradiance_Wm2", "power_kW"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
    df["station_id"] = df["station_id"].astype(str)

    df = df[
        (df["solar_irradiance_Wm2"].between(0, 1200)) &
        (df["power_kW"].between(0, 2000000))
    ]

    # معالجة أعمدة موجودة فعلاً فقط
    if "temperature_C" in df.columns:
        df["temperature_C"].fillna(25, inplace=True)
    if "humidity_%" in df.columns:
        df["humidity_%"].fillna(50.0, inplace=True)

    df = df.drop_duplicates(subset=["station_id", "timestamp"])
    return df


def feature_engineering(df):
    CAIRO_TIMEZONE = pytz.timezone('Africa/Cairo')

    # معالجة الـ timezone لأي حالة
    if hasattr(df["timestamp"].dt, "tz"):  # لو العمود كله datetime
        if df["timestamp"].dt.tz is None:
            # مش tz-aware: اعمل localize
            df["local_timestamp"] = df["timestamp"].dt.tz_localize('UTC').dt.tz_convert(CAIRO_TIMEZONE)
        else:
            # عنده timezone: فقط اعمل convert
            df["local_timestamp"] = df["timestamp"].dt.tz_convert(CAIRO_TIMEZONE)
    else:
        # fallback: فقط انسخي العمود كما هو
        df["local_timestamp"] = df["timestamp"]

    # بقية الخاصيات
    df["hour"] = df["local_timestamp"].dt.hour
    df["day_of_week"] = df["local_timestamp"].dt.day_name()
    df["time_of_day"] = df["hour"].apply(lambda h: "Day" if 6 <= h < 18 else "Night")
    df["is_valid"] = (df["solar_irradiance_Wm2"].notnull() & df["power_kW"].notnull())
    return df

def consume_and_process(max_messages=None, from_beginning=True, save_to_csv=None, show_details=False):
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=CONSUMER_GROUP,
        auto_offset_reset='earliest' if from_beginning else 'latest',
        enable_auto_commit=False,
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        consumer_timeout_ms=10000
    )

    messages = []
    count = 0
    for message in consumer:
        data = message.value
        messages.append(data)
        count += 1

        if show_details:
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            print(f"{count}: محطة {data.get('station_id', 'N/A')}, شدة الإشعاع الشمسي = {data.get('solar_irradiance_Wm2', 0)}, القدرة = {data.get('power_kW', 0)}, الوقت = {data.get('timestamp', 'N/A')}")
        
        if max_messages and count >= max_messages:
            break

    if not messages:
        print("[translate: لا توجد رسائل]")
        return

    df = pd.DataFrame(messages)
    df = validation_and_cleaning(df)
    df = feature_engineering(df)

    if save_to_csv:
        df.to_csv(save_to_csv, index=False, encoding="utf-8")
        print(f"[translate: تم حفظ البيانات بعد المعالجة في:] {save_to_csv}")

    print("[translate: عدد الرسائل بعد التنظيف والتصفية:]", len(df))
    print("[translate: أعمدة الإخراج:]", df.columns.tolist())
    print(df.head())

if __name__ == "__main__":
    consume_and_process(
        max_messages=None,
        from_beginning=True,
        save_to_csv="solar_data_processed.csv",
        show_details=False
    )
