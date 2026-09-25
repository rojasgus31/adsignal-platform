-- One row per campaign, carrying both human names.
--
-- A dimension rather than two extra columns on `fct_campaign_daily`: that table
-- has tens of thousands of rows and would repeat both strings on every one.
-- This is ~96 rows and joins once.
select
    c.campaign_id,
    c.campaign_name,
    c.advertiser_id,
    a.advertiser_name,
    c.channel,
    c.start_date,
    c.end_date
from {{ ref('stg_campaigns') }} c
inner join {{ ref('stg_advertisers') }} a on a.advertiser_id = c.advertiser_id
