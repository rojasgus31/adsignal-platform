select
    event_date,
    campaign_id,
    creative_id,
    channel,
    impressions,
    clicks,
    conversions,
    spend,
    revenue
from {{ source('raw', 'raw_events') }}
