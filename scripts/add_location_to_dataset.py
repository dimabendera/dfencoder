"""
EXAMPLE RUN:
 python3 scripts/add_location_to_dataset.py
 https://github.com/MarcCoru/locationencoder
"""
import pandas as pd

latitude_longitude_by_country = {
    ' Jamaica': (18.109581, -77.297508),
    ' Nicaragua': (12.865416, -85.207229),
    ' England': (55.378051, -3.435973),
    ' Mexico': (23.634501, -102.552784),
    ' Laos': (19.85627, 102.495496),
    ' Yugoslavia': (44.4912, 20.2539),
    ' Puerto-Rico': (18.220833, -66.590149),
    ' Cuba': (21.521757, -77.781167),
    ' India': (20.593684, 78.96288),
    ' Guatemala': (15.783471, -90.230759),
    ' Peru': (-9.189967, -75.015152),
    ' Trinadad&Tobago': (10.691803, -61.222503),
    ' Dominican-Republic': (18.735693,	-70.162651),
    ' Outlying-US(Guam-USVI-etc)': (19.2833, 166.6),
    ' ?': (None, None),
    ' Poland': (51.919438, 19.145136),
    ' Columbia': (4.00, 72.00),
    ' Iran': (32.427908, 53.688046),
    ' Vietnam': (14.058324, 108.277199),
    ' Greece': (39.074208, 21.824312),
    ' Taiwan': (23.69781, 120.960515),
    ' Philippines': (12.879721, 121.774017),
    ' Ecuador': (-1.831239, -78.183406),
    ' Hong': (22.396428, 114.109497),
    ' Portugal': (39.399872, -8.224454),
    ' Hungary': (47.162494, 19.503304),
    ' China': (35.86166, 104.195397),
    ' Holand-Netherlands': (52.132633, 5.291266),
    ' Honduras': (15.199999, -86.241905),
    ' France': (46.227638, 2.213749),
    ' El-Salvador': (13.794185, -88.89653),
    ' Japan': (36.204824, 138.252924),
    ' United-States': (37.09024, -95.712891),
    ' Scotland': (56.4396, 4.0532),
    ' Thailand': (15.870032, 100.992541),
    ' Canada': (56.130366, -106.346771),
    ' Germany': (51.165691, 10.451526),
    ' Ireland': (53.41291, -8.24389),
    ' South': (-30.559482, 22.937506),
    ' Cambodia': (12.565679, 104.990963),
    ' Haiti': (18.971187, -72.285215),
    ' Italy': (41.87194, 12.56738)
 }

if __name__ == "__main__":
    # load and look at the data
    df = pd.read_csv('./adult.csv')
    df["latitude"] = [latitude_longitude_by_country[country][0] for country in df["native-country"]]
    df["longitude"] = [latitude_longitude_by_country[country][1] for country in df["native-country"]]
    df.to_csv('./adult.csv', index=False)
