select
    id as campaign_id,
    advertiser_id,
    name as campaign_name,
    channel,
    start_date,
    end_date,
    daily_budget
from {{ source('raw', 'raw_campaigns') }}
