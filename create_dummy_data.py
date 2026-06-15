import pandas as pd
import os

os.makedirs('analysis', exist_ok=True)

df_events = pd.DataFrame({'event_date': [pd.to_datetime('2023-01-01')], 'event_count': [1]})
df_events.to_excel('analysis/cyberevents.xlsx', index=False)

df_tone = pd.DataFrame({
    'Date': [pd.to_datetime('2023-01-01')],
    'GoldsteinScale': [1.0],
    'NumArticles': [1],
    'AvgTone': [1.0]
})
df_tone.to_excel('analysis/aggregated_tone.xlsx', index=False)
