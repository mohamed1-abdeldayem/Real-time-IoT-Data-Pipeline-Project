#!/usr/bin/env python3
# [translate: قراءة بيانات محطات الرياح من Kafka ومعالجتها وتحليلها]

import json
from kafka import KafkaConsumer
import pandas as pd
from datetime import datetime, timezone
import argparse
import sys

KAFKA_BOOTSTRAP_SERVERS = ['kafka-broker-1:9092', 'kafka-broker-2:9093']
KAFKA_TOPIC = 'wind-stations'
CONSUMER_GROUP = 'wind-farm-consumer-group'

def validation_and_cleaning(df):
    # [translate: التحقق من وجود الأعمدة المطلوبة]
    required_cols = ["station_id", "timestamp", "wind_speed_mps", "farm_power_kW"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # [translate: تحويل التوقيت والصيغ]
    df["timestamp"] = df["timestamp"].str.replace('UTC', '')
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')

    # [translate: تحويل station_id إلى نص]
    df["station_id"] = df["station_id"].astype(str)

    # [translate: الفلترة القيم المنطقية]
    df = df[
        (df["wind_speed_mps"].between(0, 60)) &
        (df["air_temperature_C"].between(-50, 60)) &
        (df["farm_power_kW"].between(0, 20000000))
    ]

    # [translate: ملء القيم الافتراضية]
    df["air_pressure_hPa"].fillna(1013.25, inplace=True)
    df["humidity_percent"].fillna(50.0, inplace=True)

    # [translate: حذف التكرارات]
    df = df.drop_duplicates(subset=["station_id", "timestamp"])

    return df


def feature_engineering(df):
    # [translate: وقت محلي Africa/Cairo]
    try:
        import pytz
        cairo = pytz.timezone('Africa/Cairo')
        df["local_timestamp"] = df["timestamp"].dt.tz_localize('UTC').dt.tz_convert(cairo)
    except ImportError:
        df["local_timestamp"] = df["timestamp"]  # بدون تحويل للوقت المحلي إذا لم يتوفر pytz

    df["hour"] = df["local_timestamp"].dt.hour
    df["day_of_week"] = df["local_timestamp"].dt.day_name()

    df["time_of_day"] = df["hour"].apply(lambda h: "Day" if 6 <= h < 18 else "Night")

    # [translate: حساب كثافة القدرة من سرعة وكثافة الهواء]
    df["wind_power_density"] = 0.5 * df["air_density_kgm3"] * (df["wind_speed_mps"] ** 3)

    df["is_valid"] = (df["wind_speed_mps"].notnull() & df["farm_power_kW"].notnull())

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
            print(f"{count}: محطة {data.get('station_id', 'N/A')}, سرعة الرياح = {data.get('wind_speed_mps', 0)}, القدرة = {data.get('farm_power_kW', 0)}, الوقت = {data.get('timestamp', 'N/A')}")
        
        if max_messages and count >= max_messages:
            break

    if not messages:
        print("[translate: لا توجد رسائل]")
        return

    df = pd.DataFrame(messages)

    # [translate: معالجة وتنظيف البيانات]
    df = validation_and_cleaning(df)
    df = feature_engineering(df)

    # [translate: حفظ البيانات في ملف CSV إذا طُلب]
    if save_to_csv:
        df.to_csv(save_to_csv, index=False, encoding="utf-8")
        print(f"[translate: تم حفظ البيانات بعد المعالجة في:] {save_to_csv}")

    print("[translate: عدد الرسائل بعد التنظيف والتصفية:]", len(df))
    print("[translate: أعمدة الإخراج:]", df.columns.tolist())
    print(df.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="[translate: قراءة بيانات ومحطات الرياح من Kafka ومعالجتها بالكامل]"
    )
    parser.add_argument('-n', '--num-messages', type=int, default=None, help="[translate: عدد الرسائل المطلوبة (افتراضي: كل الرسائل)]")
    parser.add_argument('--from-beginning', action='store_true', help="[translate: البدء من أول رسالة]")
    parser.add_argument('--csv', type=str, default=None, help="[translate: حفظ البيانات المعالجة في ملف CSV]")
    parser.add_argument('--details', action='store_true', help="[translate: عرض تفاصيل كل رسالة]")
    args = parser.parse_args()

    consume_and_process(
        max_messages=args.num_messages,
        from_beginning=args.from_beginning,
        save_to_csv=args.csv,
        show_details=args.details
    )
