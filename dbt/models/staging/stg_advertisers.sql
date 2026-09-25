select
    id as advertiser_id,
    name as advertiser_name
from {{ source('raw', 'raw_advertisers') }}
