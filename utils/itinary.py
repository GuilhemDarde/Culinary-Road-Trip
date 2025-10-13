import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def pick_restaurants_for_city(df_city, cuisines, price_level, n=2):
    data = df_city.copy()
    if cuisines:
        data = data[data['cuisine_list'].apply(lambda L:any(c in L for c in cuisines))]
    if price_level != "Any":
        if price_level == "Economy":
            data = data[data['price_num'] <= 1.5]
        elif price_level == "Moderate":
            data = data[(data['price_num']>1.5) & (data['price_num']<=3)]
        else:
            data = data[data['price_num']>3]
    if data.empty: data = df_city
    return data.sort_values('score_final',ascending=False).head(n)

def build_itinerary(route_points, df, days, cuisines, price_level):
    rows, restos = [], []
    if not route_points: return pd.DataFrame(), pd.DataFrame()
    n_cities = len(route_points)
    base, rem = days//n_cities, days%n_cities
    per_city = [base+(1 if i<rem else 0) for i in range(n_cities)]
    current = 1
    for i,(city,lat,lon,country) in enumerate(route_points):
        sub = df[(df['city']==city)&(df['country']==country)]
        stay = per_city[i]
        for d in range(stay):
            picks = pick_restaurants_for_city(sub,cuisines,price_level,2)
            lunch = picks.iloc[0] if len(picks)>0 else None
            dinner = picks.iloc[1] if len(picks)>1 else None
            rows.append({
                'day':current,'city':city,'country':country,
                'lunch_name':lunch['restaurant_name'] if lunch is not None else '',
                'dinner_name':dinner['restaurant_name'] if dinner is not None else ''
            })
            for tag,row in [('Lunch',lunch),('Dinner',dinner)]:
                if row is not None:
                    restos.append({
                        'type':tag,'restaurant_name':row['restaurant_name'],'city':city,
                        'country':country,'latitude':row['latitude'],'longitude':row['longitude'],
                        'rating':row.get('rating',np.nan),'price_level':row.get('price_level',''),
                        'cuisine':row.get('cuisine','')
                    })
            current+=1
            if current>days:break
        if current>days:break
    return pd.DataFrame(rows), pd.DataFrame(restos)

def make_ics(itinerary_df, start_date=None):
    if start_date is None: start_date = datetime.today().date()
    ics = ["BEGIN:VCALENDAR","VERSION:2.0"]
    for _,row in itinerary_df.iterrows():
        for when,title in [('12:30','Lunch'),('19:30','Dinner')]:
            h,m=map(int,when.split(':'))
            start=datetime.combine(start_date+timedelta(days=int(row['day'])-1),datetime.min.time())+timedelta(hours=h,minutes=m)
            end=start+timedelta(hours=1,minutes=30)
            ics += [
                "BEGIN:VEVENT",
                f"DTSTART:{start.strftime('%Y%m%dT%H%M%S')}",
                f"DTEND:{end.strftime('%Y%m%dT%H%M%S')}",
                f"SUMMARY:{title} — {row['city']} — {row[title.lower()+'_name']}",
                "END:VEVENT"
            ]
    ics.append("END:VCALENDAR")
    return "\n".join(ics)