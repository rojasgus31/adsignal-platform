with daily as (

    select
        e.event_date,
        e.campaign_id,
        e.channel,
        sum(e.impressions) as impressions,
        sum(e.clicks)      as clicks,
        sum(e.conversions) as conversions,
        sum(e.spend)       as spend,
        sum(e.revenue)     as revenue
    from {{ ref('stg_events') }} as e
    group by 1, 2, 3

),

labelled as (

    select distinct
        event_date,
        campaign_id,
        cause    as anomaly_cause,
        severity as anomaly_severity
    from {{ source('raw', 'raw_anomaly_labels') }}

)

select
    d.event_date,
    d.campaign_id,
    c.advertiser_id,
    d.channel,
    d.impressions,
    d.clicks,
    d.conversions,
    d.spend,
    d.revenue,

    -- Rate metrics are NULL when their denominator is zero. A zero would
    -- read as "this campaign converts at 0%" rather than "unknowable".
    d.clicks      / nullif(d.impressions, 0)          as ctr,
    d.spend       / nullif(d.clicks, 0)               as cpc,
    d.spend       / nullif(d.impressions, 0) * 1000.0 as cpm,
    d.spend       / nullif(d.conversions, 0)          as cpa,
    d.revenue     / nullif(d.spend, 0)                as roas,

    l.anomaly_cause,
    l.anomaly_severity,
    l.anomaly_cause is not null as is_anomaly

from daily as d
inner join {{ ref('stg_campaigns') }} as c on c.campaign_id = d.campaign_id
left join labelled as l
    on  l.campaign_id = d.campaign_id
    and l.event_date  = d.event_date
