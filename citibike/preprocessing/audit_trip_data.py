#!/usr/bin/env python3
"""Day 1 Citi Bike audit. Python 3.10+. No optimizer required.

Examples:
 python -m citibike.preprocessing.audit_trip_data --zip ./202604-citibike-tripdata.zip --output ./results
 python -m citibike.preprocessing.audit_trip_data --zip ./202604-citibike-tripdata.zip --metadata ./station_information.json --output ./results

Only trip-data-derived statistics are historical. Live metadata capacity is tagged
as a CURRENT snapshot and is never represented as historical capacity.
"""
from __future__ import annotations
import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLS = ['ride_id','rideable_type','started_at','ended_at','start_station_name',
        'start_station_id','end_station_name','end_station_id',
        'start_lat','start_lng','end_lat','end_lng']
DATE_FMT = '%Y-%m-%d'


def distance_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1,p2=np.radians(lat1),np.radians(lat2)
    dp=p2-p1
    dl=np.radians(lon2-lon1)
    a=np.sin(dp/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*r*np.arcsin(np.sqrt(np.clip(a,0,1)))


def load_metadata(path):
    if not path:
        return pd.DataFrame(columns=['station_id','capacity','metadata_lat','metadata_lon','metadata_name','gbfs_station_id'])
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    stations=data.get('data',{}).get('stations',[])
    rec=[]
    for s in stations:
        sid=str(s.get('short_name','')).strip()  # Trip station_id uses GBFS short_name
        if not sid: continue
        name=s.get('name','')
        if isinstance(name,list):
            name=next((item.get('text','') for item in name if isinstance(item,dict) and item.get('text')), '')
        rec.append({'station_id':sid,'capacity':s.get('capacity'),
                    'metadata_lat':s.get('lat'),'metadata_lon':s.get('lon'),
                    'metadata_name':name,'gbfs_station_id':str(s.get('station_id','')).strip()})
    if not rec: raise ValueError('Metadata has no station records')
    return pd.DataFrame(rec).drop_duplicates('station_id')


def scan(zippath, chunksize, month):
    quality=Counter(); station=defaultdict(lambda: {'start':0,'end':0,'days':set(),
                     'lat_sum':0.,'lon_sum':0.,'coords_n':0,'name':''})
    hourly=Counter(); daily_hour=Counter(); daily_net=Counter(); pairs=Counter(); monthly_types=Counter()
    seen=set(); csv_files=[]; valid_time_min=None; valid_time_max=None
    with ZipFile(zippath) as z:
        names=[n for n in z.namelist() if n.lower().endswith('.csv') and not n.startswith('__MACOSX/')]
        if not names: raise ValueError('ZIP contains no CSV files')
        for name in sorted(names):
            csv_files.append(name)
            with z.open(name) as f:
                for chunk in pd.read_csv(f, chunksize=chunksize, low_memory=False, dtype={
                        'ride_id':'string','start_station_id':'string','end_station_id':'string'}):
                    missing=[c for c in COLS if c not in chunk.columns]
                    if missing: raise ValueError(f'{name} missing expected columns: {missing}')
                    quality['raw_rows']+=len(chunk)
                    quality['raw_missing_start_id']+=int(chunk.start_station_id.isna().sum())
                    quality['raw_missing_end_id']+=int(chunk.end_station_id.isna().sum())
                    quality['raw_missing_ride_id']+=int(chunk.ride_id.isna().sum())
                    st=pd.to_datetime(chunk.started_at,errors='coerce')
                    et=pd.to_datetime(chunk.ended_at,errors='coerce')
                    ok=st.notna() & et.notna()
                    quality['bad_datetime']+=int((~ok).sum())
                    duration=(et-st).dt.total_seconds()
                    quality['invalid_duration']+=int(((duration<=0)|(duration>24*3600)).fillna(False).sum())
                    ok &= (duration>0)&(duration<=24*3600)
                    if month:
                        period=st.dt.strftime('%Y-%m')
                        outside=(period!=month).fillna(True)
                        quality['outside_month']+=int((ok & outside).sum())
                        ok &= ~outside
                    ids=chunk.ride_id.fillna('').astype(str).str.strip()
                    dupe=ids.duplicated(keep='first') | ids.isin(seen)
                    nonempty=ids.ne('')
                    # add even invalid row ids so duplicate checks apply to all source records
                    seen.update(ids[nonempty].tolist())
                    quality['duplicate_ride_id']+=int((dupe & nonempty).sum())
                    ok &= ~dupe & nonempty
                    ss=chunk.start_station_id.fillna('').astype(str).str.strip()
                    ee=chunk.end_station_id.fillna('').astype(str).str.strip()
                    ok &= ss.ne('') & ee.ne('')
                    for column in ['start_lat','start_lng','end_lat','end_lng']:
                        chunk[column]=pd.to_numeric(chunk[column],errors='coerce')
                    good_geo=chunk.start_lat.between(-90,90) & chunk.end_lat.between(-90,90) & chunk.start_lng.between(-180,180) & chunk.end_lng.between(-180,180)
                    quality['invalid_geo']+=int((ok & ~good_geo).sum())
                    ok &= good_geo
                    valid=chunk.loc[ok].copy()
                    if valid.empty: continue
                    valid['st']=st.loc[ok].values
                    valid['et']=et.loc[ok].values
                    valid['ss']=ss.loc[ok].values
                    valid['ee']=ee.loc[ok].values
                    quality['valid_rows']+=len(valid)
                    quality['cross_date_trips']+=int((valid.st.dt.date!=valid.et.dt.date).sum())
                    mn,mx=valid.st.min(),valid.et.max()
                    valid_time_min=mn if valid_time_min is None else min(mn,valid_time_min)
                    valid_time_max=mx if valid_time_max is None else max(mx,valid_time_max)
                    monthly_types.update(valid.rideable_type.fillna('unknown').astype(str).value_counts().to_dict())
                    for side, idcol, dtcol, latcol, lngcol in [('start','ss','st','start_lat','start_lng'),('end','ee','et','end_lat','end_lng')]:
                        gb=valid.groupby(idcol,sort=False)
                        for sid,g in gb:
                            s=station[sid]
                            s[side]+=len(g)
                            s['days'].update(g[dtcol].dt.strftime(DATE_FMT).unique().tolist())
                            s['lat_sum']+=float(g[latcol].sum())
                            s['lon_sum']+=float(g[lngcol].sum())
                            s['coords_n']+=len(g)
                            if not s['name']:
                                x=g['start_station_name' if side=='start' else 'end_station_name'].dropna()
                                if len(x): s['name']=str(x.iloc[0])
                        keys=list(zip(valid[idcol].astype(str),valid[dtcol].dt.strftime(DATE_FMT),valid[dtcol].dt.hour))
                        hourly.update((sid,day,h,side) for sid,day,h in keys)
                        daily_hour.update((day,h,side) for sid,day,h in keys)
                        daily_net.update((sid,day,h,1 if side=='end' else -1) for sid,day,h in keys)
                    pairs.update(zip(valid.ss.astype(str), valid.ee.astype(str)))
    if quality['valid_rows']==0: raise ValueError('No valid trips after cleaning')
    rows=[]
    for sid,s in station.items():
        n=s['coords_n']
        rows.append({'station_id':sid,'station_name':s['name'],'latitude':s['lat_sum']/n,'longitude':s['lon_sum']/n,
                     'active_days':len(s['days']),'departures':s['start'],'arrivals':s['end'],
                     'total_activity':s['start']+s['end']})
    stations=pd.DataFrame(rows)
    hrs=pd.DataFrame([{'station_id':sid,'date':day,'hour':hour,'type':kind,'count':val}
                      for (sid,day,hour,kind),val in hourly.items()])
    od=pd.DataFrame([{'start_station_id':a,'end_station_id':b,'trips':n} for (a,b),n in pairs.items()])
    report={'quality':dict(quality),'csv_members':csv_files,'valid_time_min':str(valid_time_min),
            'valid_time_max':str(valid_time_max),'rideable_types':dict(monthly_types),'distinct_stations':len(stations),
            'note':'Historical rides are observed fulfilled trips, not all latent customer demand.'}
    return stations,hrs,od,report


def candidates(stations, hrs, od, k, n_clusters=5, min_days=12):
    # Restrict selection statistics to initial training window (first 20 days).
    first=pd.to_datetime(hrs.date.min()); train_end=first+pd.Timedelta(days=19)
    h=hrs[pd.to_datetime(hrs.date)<=train_end].copy()
    sums=h.groupby(['station_id','type'])['count'].sum().unstack(fill_value=0)
    activity=sums.sum(axis=1)
    active=h.groupby('station_id').date.nunique()
    base=stations.set_index('station_id').copy()
    base['train_activity']=activity.reindex(base.index,fill_value=0)
    base['train_days']=active.reindex(base.index,fill_value=0)
    eligible=base[(base.train_days>=min_days)&(base.train_activity>0)].copy()
    if len(eligible)<k:
        raise ValueError(f'Only {len(eligible)} eligible stations for requested k={k}; reduce --stations or --min-days')
    # Seed from top activity quantile, then greedily grow a compact candidate around it.
    seeds=eligible.sort_values('train_activity',ascending=False).head(max(n_clusters*5,n_clusters)).index.tolist()
    picks=[]
    for seed in seeds:
        ss=eligible.loc[seed]
        dist=distance_km(float(ss.latitude),float(ss.longitude),eligible.latitude.to_numpy(),eligible.longitude.to_numpy())
        local=eligible.assign(dist_km=dist).sort_values(['dist_km','train_activity'],ascending=[True,False]).head(k)
        nodes=set(local.index)
        internal=int(od[od.start_station_id.isin(nodes)&od.end_station_id.isin(nodes)].trips.sum())
        outgoing=int(od[od.start_station_id.isin(nodes)&~od.end_station_id.isin(nodes)].trips.sum())
        incoming=int(od[~od.start_station_id.isin(nodes)&od.end_station_id.isin(nodes)].trips.sum())
        relevant=internal+incoming+outgoing
        frac=(incoming+outgoing)/relevant if relevant else np.nan
        # Calculate training time-series cumulative imbalances per station
        hh=h[h.station_id.isin(nodes)]
        piv=hh.pivot_table(index=['station_id','date','hour'],columns='type',values='count',aggfunc='sum',fill_value=0)
        piv['net']=piv.get('end',0)-piv.get('start',0)
        avg_range=float(piv.groupby(level=['station_id','date']).net.apply(lambda z: (z.cumsum().max()-z.cumsum().min())).mean()) if not piv.empty else 0.
        picks.append({'seed_station_id':seed,'station_count':k,'max_radius_km':round(float(local.dist_km.max()),3),
                      'mean_activity_train':round(float(local.train_activity.mean()),2),
                      'cumulative_net_range_mean':round(avg_range,2),
                      'internal_trips_full_month':internal,'incoming_trips_full_month':incoming,
                      'outgoing_trips_full_month':outgoing,'cross_boundary_share_full_month':round(float(frac),4),
                      '_members':sorted(nodes)})
    # Deduplicate identical candidate member sets
    result=[]; seen=set()
    for p in picks:
        key=tuple(p['_members'])
        if key in seen: continue
        seen.add(key); result.append(p)
        if len(result)>=n_clusters:break
    # Important: full-month boundary metrics are DIAGNOSTIC only, never used for cluster selection.
    return result, h


def plots(stations, hrs, candidates_, out):
    out.mkdir(parents=True,exist_ok=True)
    h=hrs.copy(); h['hour']=pd.to_numeric(h.hour)
    counts=h.groupby(['hour','type'])['count'].sum().unstack(fill_value=0).reindex(range(24),fill_value=0)
    fig,ax=plt.subplots(figsize=(9,4.5))
    for c,label in [('start','Departures'),('end','Arrivals')]:
        if c in counts:ax.plot(counts.index,counts[c],label=label)
    ax.set(xlabel='Hour (local time as published)',ylabel='Trip events',title='Observed hourly trip activity')
    ax.legend(loc='upper left');fig.tight_layout();fig.savefig(out/'01_hourly_demand.png',dpi=160);plt.close(fig)
    ids=set(candidates_[0]['_members'])
    selected=stations[stations.station_id.isin(ids)]
    fig,ax=plt.subplots(figsize=(8,6))
    ax.scatter(stations.longitude,stations.latitude,s=5,alpha=.23,label='All active stations')
    ax.scatter(selected.longitude,selected.latitude,s=28,label='Selected candidate')
    ax.set(xlabel='Longitude',ylabel='Latitude',title='Candidate spatial cluster (not final selection)')
    ax.legend(loc='upper left');fig.tight_layout();fig.savefig(out/'02_candidate_map.png',dpi=160);plt.close(fig)
    p=hrs[hrs.station_id.isin(ids)].pivot_table(index=['station_id','date','hour'],columns='type',values='count',aggfunc='sum',fill_value=0)
    p['net']=p.get('end',0)-p.get('start',0)
    p=p.reset_index().sort_values(['station_id','date','hour'])
    totals=p.groupby('station_id').net.sum().abs().sort_values(ascending=False).head(8).index
    fig,ax=plt.subplots(figsize=(9,4.5))
    # profile average across active observed days only: diagnostic, not physical stock
    for sid in totals:
        q=p[p.station_id==sid].groupby('hour').net.mean().reindex(range(24),fill_value=0).cumsum()
        ax.plot(q.index,q.values,label=str(sid))
    ax.set(xlabel='Hour',ylabel='Cumulative mean hourly net inflow',title='Candidate cluster: net-flow profiles (not inventory)')
    ax.legend(loc='upper left',fontsize=7,ncol=2);fig.tight_layout();fig.savefig(out/'03_net_flow_profiles.png',dpi=160);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description='Citi Bike Day 1 data audit')
    ap.add_argument('--zip',required=True,type=Path,help='Local trip history ZIP')
    ap.add_argument('--metadata',type=Path,default=None,help='Local GBFS station_information.json snapshot')
    ap.add_argument('--output',type=Path,default=Path('audit_results'))
    ap.add_argument('--month',default='2026-04',help='Trip START month YYYY-MM, or empty to keep all')
    ap.add_argument('--stations',type=int,default=25)
    ap.add_argument('--min-days',type=int,default=12)
    ap.add_argument('--chunk-size',type=int,default=100000)
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    stations,hrs,od,report=scan(args.zip,args.chunk_size,args.month)
    metadata=load_metadata(args.metadata)
    stations=stations.merge(metadata,how='left',on='station_id',validate='one_to_one')
    stations['capacity_source']=np.select([stations.capacity.gt(0),stations.capacity.notna()],['CURRENT_GBFS_POSITIVE_NOT_HISTORICAL','CURRENT_GBFS_NONPOSITIVE_NOT_HISTORICAL'],default='UNAVAILABLE')
    report['metadata_matched_stations']=int(stations.capacity.notna().sum())
    report['metadata_match_fraction']=round(float(stations.capacity.notna().mean()),4)
    report['metadata_positive_capacity_stations']=int(stations.capacity.gt(0).sum())
    report['metadata_timestamp_utc']=datetime.now(timezone.utc).isoformat() if args.metadata else None
    report['metadata_warning']='A current GBFS snapshot does not establish station capacity in the trip month.'
    clusters,train_h=candidates(stations,hrs,od,args.stations,min_days=args.min_days)
    report['selection_warning']='Candidate clusters use only first 20 calendar days for selection. Full-month boundary figures are diagnostic, not for ranking.'
    stations.sort_values('total_activity',ascending=False).to_csv(args.output/'station_registry.csv',index=False)
    hrs.to_csv(args.output/'station_flow_summary.csv',index=False)
    od.sort_values('trips',ascending=False).to_csv(args.output/'station_od_counts.csv',index=False)
    pd.DataFrame([{**{key:v for key,v in row.items() if key!='_members'},'station_ids':';'.join(row['_members'])} for row in clusters]).to_csv(args.output/'candidate_clusters.csv',index=False)
    (args.output/'data_quality_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    plots(stations,hrs,clusters,args.output)
    summary=f'''# Day 1 Audit Summary\n\n- Valid observed trips: {report['quality']['valid_rows']:,}\n- Distinct stations: {report['distinct_stations']:,}\n- Capacity metadata match: {report['metadata_matched_stations']}/{report['distinct_stations']}\n- First cluster seed: {clusters[0]['seed_station_id']}\n- First cluster cross-boundary share (full-month diagnostic): {clusters[0]['cross_boundary_share_full_month']:.1%}\n\n## Important limitations\n\n- Observed completed trips do not include unserved demand.\n- GBFS capacities are current snapshot information unless an archived matching snapshot is provided.\n- Initial station inventory and operational costs are not known.\n- Cross-boundary flows must be modeled; the candidate cluster is not a closed system.\n- This stage does not simulate stocks or solve an optimization problem.\n- Hourly timestamps are treated as published local wall times; April avoids US DST transition.\n\n## Next decision\n\nReview metadata coverage and candidate clusters before choosing the final subnetwork and modeling initial inventory.\n'''
    (args.output/'audit_summary.md').write_text(summary,encoding='utf-8')
    print(summary)
    print('Output:',args.output.resolve())

if __name__=='__main__':main()
