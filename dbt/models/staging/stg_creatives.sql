select
    id as creative_id,
    campaign_id,
    headline,
    format
from {{ source('raw', 'raw_creatives') }}
