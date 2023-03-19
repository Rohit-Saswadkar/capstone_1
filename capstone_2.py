# all CSV data
import numpy
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import seaborn as sns

df=pd.read_csv(r"C:\Users\saswa\OneDrive\Documents\pandas data\data\capstone 1\hotel bookings xx.csv")
print(df)

print(len(df))

# check information about the dataframe
print(df.describe())
print(df.shape)

# NOW WE ARE CHECKING THAT ARE THERE NULL VALUES IN DATAFRAME
print(df.isnull().sum())

# data cleaning and evaluate missing values for evaluation
# create a function to find the percentage of null values
def percent_null(df):
    per=df.isnull().sum()/len(df)*100
    print('per')
    print(per)
    perm=per[per>0].sort_values(ascending=False)
    return perm
missing_values=round(percent_null(df),2)
print(missing_values)
# plot the chart as per percentage of missing values
g=missing_values.plot(figsize=(10,5),kind='bar')
print(plt.title('Null value percentage', fontsize=20))
print(plt.ylabel('PERCENTAGE',fontsize=17))
print(plt.xlabel('Columns Null values',fontsize=17))
# print(plt.show())

# as company column has many null values so we are going to drop it
print(df.shape)
df.drop(['company'],axis=1,inplace=True)
print(df.shape)
df.country.fillna('no data',inplace=True)
# print(df.country)
# # now we are filling the null values of agent by 'no'
df.agent.fillna('no agent',inplace=True)
print(df.agent)

df.agent.fillna('agent xxx',inplace=True)
print(df.agent)
print(df.children)
df.children.fillna(0,inplace=True)
df['children'].astype('int64')
print(df.children)

# check if any null value present in dataset
print(df.isnull().sum())

# lets check duplicate data, if available , we have to remove it
print(df.info)
print(df.duplicated().sum())
print(df.drop_duplicates(inplace=True))
print(df.info)

# now we create a column name total stay
df['total_guests']=df['adults']+df['children']+df['babies']

print(type(df['total_guests']))
# convert total gust into integer
df['total_guests']=df['total_guests'].astype('int64')
df['total_stay']=df['stays_in_weekend_nights']
print(df.total_stay)
# 1] get booking percentage by hotel
#to seprate both city and resort booking we use groupby method
print('1] start')
grouped_by_hotel = df.groupby('hotel')
print('1')
print(grouped_by_hotel.size())
print(df.shape)
print('2')
d1 = pd.DataFrame((grouped_by_hotel.size()/df.shape[0])*100).reset_index().rename(columns = {0:'Booking %'})
print(d1)
plt.figure(figsize = (5,7))
sns.barplot(x = d1['hotel'], y = d1['Booking %'] )
# plt.show()
print('1] end')

# 2] % of sum of booking that arent cancelled by month
print('2] start')
#adding booking cancelation and confrimed and groupby for city & Resort
sum_of_booking_arnt_cancelled=(df.groupby('hotel')['previous_bookings_not_canceled'].sum())
x=pd.DataFrame(sum_of_booking_arnt_cancelled).reset_index()
print(x)
plt.figure(figsize = (5,7))
sns.barplot(x = x['hotel'], y = x['previous_bookings_not_canceled'] )
# plt.show()
print('2] end')
# 3] stays in weekday nights by both hotels
#count total stays weekday nights for City & Resort hotel
print('3] start')
stays_by_weeknights=df['stays_in_week_nights'].value_counts()
print(stays_by_weeknights)

plt.figure(figsize=(10,5))
sns.countplot(data = df, x = 'stays_in_week_nights' ).set_title('Number of stays on weekday night' , fontsize = 20)
plt.xlabel('No. of stays on week nights', fontsize = 15)
plt.ylabel('Total Count ',fontsize = 15)
print('3] end')

#4] stays in weekend nights by both hotels
print('4] start')
#similarly counts total stays weekend nights for both type hotels
stays_by_weekend_nights=df['stays_in_weekend_nights'].value_counts()
print(stays_by_weekend_nights)

plt.figure(figsize=(8,5))
sns.countplot(data = df, x = 'stays_in_weekend_nights').set_title('Number of stays on weekend nights', fontsize = 20)
plt.xlabel('No. of stays on weekend nights', fontsize = 15)
plt.ylabel('Total Count ',fontsize = 15)
print('4]end')
# 5] sum of repeated guests by month wise
print('5] start')
#count total no of repeated custmors by month & index with new dataframe
#use sum firstly and after use reset index function
rep_guests=df.groupby(by=['hotel','arrival_date_month'])['is_repeated_guest'].sum()
y=pd.DataFrame(rep_guests).reset_index()
print(y)
y.plot(figsize=(10,5),kind='bar')
plt.title('sum of repeated guests by month wise',fontsize=20)
plt.ylabel('repeated guests',fontsize=17)
plt.xlabel('hotel',fontsize=17)
# plt.show()
print('5] end')

# 6]calculate cancellation by months
print('6] start')
#group by total cancelation for both hotels
#take minimum one cancelation 
cancelled_data=df[df['is_canceled']==1]
cancelled=cancelled_data.groupby('arrival_date_month')
x=pd.DataFrame(cancelled.size()).rename(columns={0:'total_cancelled_bookings'})

grouped_by_hotel=df.groupby('arrival_date_month')
total_booking=grouped_by_hotel.size()
y=pd.DataFrame(total_booking).rename(columns={0: 'total_bookings'})
z=pd.concat([x,y],axis=1)
print(z)

z['cancel_%']=round((z['total_cancelled_bookings']/z['total_bookings'])*100,2)
plt.figure(figsize=(10,5))
sns.barplot(x=z.index, y=z['cancel_%']).set_title("Cancelation Based on Months" , fontsize = 20)
# plt.show()
print('6] end')

# 7] number of repeated guest type
print('7] start')
#count totlal repeted guest for that use value count function and seprated with groupby funtion
customer=df.groupby(by=['hotel','is_repeated_guest'])['customer_type'].value_counts()
print(customer)

plt.figure(figsize=(5,7))
sns.countplot(data = df, x = 'customer_type').set_title('number of repeated guest type ', fontsize = 20)
plt.xlabel('customer types', fontsize = 15)
plt.ylabel('Repeated Guests',fontsize = 15)
# plt.show()
print('7] end')

# 8] no of adults by hotel
print('8] start')
#similarly just count total no of adults 
#use value count function
no_of_adults=df.groupby('hotel')['adults'].value_counts()
print(no_of_adults)
plt.rcParams['figure.figsize'] = (12,5)
sns.countplot(data = df, x = 'adults', hue = 'hotel').set_title("Number of adults", fontsize = 20)
plt.ylabel('Count of adults',fontsize = 15)
plt.xlabel('Total no. of adults travelled',fontsize = 15)
# plt.show()
print('8] end')

# 9] Hotel having High Cancelation Rate

print('9] start')
#for that take minimum one canceled booking weekely basis
cancelled_data_over_week_no=df[df['is_canceled']==1]
cancel=cancelled_data_over_week_no.groupby('hotel')
x=pd.DataFrame(cancel.size()).rename(columns={0:'total_cancelled_bookings'})

grouped_hotel=df.groupby('hotel')
total_booking_done=grouped_hotel.size()
y=pd.DataFrame(total_booking_done).rename(columns={0: 'total_bookings'})
z=pd.concat([x,y], axis=1)

z['cancel%'] = round((z['total_cancelled_bookings']/z['total_bookings'])*100,2)
print(z)

plt.figure(figsize = (5,7))
sns.barplot(x=z.index,y=z['cancel%']).set_title("Hotel having High Cancelation Rate",fontsize = 20)
# plt.show()
print('9] end')
# 10] which hotels has preferred for breakfast and dinner
print('10] start')
#take data of hotel who has atleast avilabilty of one time breakfast
bb=df[df['meal']=='BB']
bbc=bb.groupby('hotel')
df_bb=pd.DataFrame(bbc.size()).rename(columns={0:'BB'})
print(df_bb['BB'])

#take data of hotel who has atleast avilabilty of one time lunch
fb=df[df['meal']=='FB']
fbc=fb.groupby('hotel')
fbc_df=pd.DataFrame(fbc.size()).rename(columns={0:'FB'})
print(fbc_df['FB'])

#take data of hotel who has atleast avilabilty of one time dinner
hb=df[df['meal']=="HB"]
hbc=hb.groupby('hotel')
hbc_df=pd.DataFrame(hbc.size()).rename(columns={0:'HB'})
print(hbc_df['HB'])
concat_meal=pd.concat([df_bb,fbc_df['FB'],hbc_df['HB']],axis=1)
print(concat_meal)

concat_meal.plot(figsize=(10,5),kind='bar',log=False)
plt.title('most prefered meals', fontsize=20)
plt.ylabel('number of meals',fontsize=17)
plt.xlabel('types of meals',fontsize=17,)
# plt.show()

print('10] end')

# 11] get duration of stays in each Hotel
print('11] start')
#count total no of stays min one day and maximum 20 days
not_canceled=df[df['is_canceled']==0]
x=not_canceled[not_canceled['total_stay']<20]

plt.figure(figsize=(15,7))
sns.countplot(x=x['total_stay'],hue=x['hotel'])
# plt.show()

print('11] end')
# 12] get number of arrival per year 
#count all custmors who books hotels room in one year

print(' 12] start')
x=df['arrival_date_year'].value_counts()
print(x)
l=['hotel','arrival_date_year','arrival_date_month','arrival_date_day_of_month' ]
period_arrival=df[l]
plt.figure(figsize=(7,7))
sns.countplot(data=period_arrival,x='arrival_date_year',hue='hotel').set_title('Number of arrival per Year',fontsize=20)
# plt.show()
print(len(df.groupby('country').count()))
print('12] end')

# 13] get arrival date month
print('13] start')
#count all custmors who books hotels room in one month

arr_per_month=df['arrival_date_month'].value_counts()
print(arr_per_month)
sns.countplot(data = period_arrival, x = 'arrival_date_month', hue = 'hotel').set_title('Number of arrival per month',fontsize=20)
plt.xlabel('Arrival Month',fontsize = 15)
plt.ylabel('Total Count', fontsize = 15)
# plt.show()
print('xp2')
print('13] end')

# 14] count arrival per day
print('14] start')
#count all custmors who books hotels room in one day
arr_per_day=df['arrival_date_day_of_month'].value_counts()
print(arr_per_day)

plt.figure(figsize=(15,5))
sns.countplot(data = period_arrival, x = 'arrival_date_day_of_month', hue = 'hotel').set_title('Number of arrival per day', fontsize = 20)
plt.xlabel('Arrival Day', fontsize = 15)
plt.ylabel('Total Count ',fontsize = 15)
# plt.show()
print('14] end')

# 15] average adr per month by hotel
print('15] start')
#calculate adr for each hotel using total arrival for city or resort divided with total arrival of all types hotel multiply by hundred
percentage_col=df.groupby(by=['hotel','arrival_date_month'])['adr'].sum()/[df[df['hotel']=='Resort Hotel']['adr'].sum()]*100
print(percentage_col)
percentage_col.plot(figsize=(10,7),kind='bar',log=False)
plt.title('% of ADR by month', fontsize=20)
plt.ylabel('percentage',fontsize=17)
plt.xlabel('months by hotel',fontsize=17,)
# plt.show()
print('15] end')

# 16] parkings required per month
print('16] start')
#take atleast one parking for hotels using normalize function 
x=df.groupby('hotel')['required_car_parking_spaces'].value_counts(normalize=False)
plt.figure(figsize=(15,5))
sns.countplot(data = df, x = 'required_car_parking_spaces', hue = 'hotel').set_title('required_car_parking_spaces', fontsize = 20)
plt.xlabel('required_car_parking_spaces', fontsize = 15)
plt.ylabel('Total Count ',fontsize = 15)
# plt.show()
print(x)
print('16] end')

# 17] which market segment has more customers by hotel
print('17] start')
#seprated total no of custmors using value count function
mark_seg=df.groupby('hotel')['market_segment'].value_counts()
print(mark_seg)

x=df[['hotel','market_segment']]
plt.figure(figsize=(7,7))
sns.countplot(data =x, x = 'market_segment', hue = 'hotel').set_title('customers by market segment',fontsize=20)
# plt.show()
print('17] end')

# 18] get types of distribution channel
print('18] start')
#seprated total no of booking which comes from diffrent types of distribution network using value count function
dist_channel=df['distribution_channel'].value_counts()
print(dist_channel)
plt.figure(figsize=(12,5))
sns.countplot(data = df, x = 'distribution_channel').set_title('Types of distribution channel', fontsize = 20)
plt.xlabel('Type of distribution channel', fontsize = 15)
plt.ylabel('Total Count ',fontsize = 15)
# plt.show()
print('18] end')

#19] get top 5 countries by city and resort hotels
print('19] start')
#first counts total booking 
#second sort total booking with country wise
#fianally ascending the data high to low counts and take top 5 no of booking
hotel_country = df.groupby('country',as_index=True)['hotel'].value_counts()
hotel_country = hotel_country.sort_values(ascending=False)
hotel_country_df = pd.DataFrame(hotel_country)
hotel_country_df=hotel_country_df.head(10)

hotel_country_df.plot(kind='pie',subplots=True)
# plt.show()
print('19] end')

# 20] most preffered type of deposit
print('20] start')
#count total booking and seprated with deposite type using value count
dep_type=df.groupby('hotel')['deposit_type'].value_counts()
print(dep_type)
plt.figure(figsize=(12,5))
sns.countplot(data = df, x = 'deposit_type').set_title('Types of deposits', fontsize = 20)
plt.ylabel('Count of Bookings',fontsize = 15)
plt.xlabel('Deposit Types',fontsize = 15)
# plt.show()
print('20] end')

# 21] check reservation status by hotel
print('21] start')
#seprated reservation status using value count function
res_stat=df.groupby('hotel')['reservation_status'].value_counts()
print(res_stat)
x=df[['hotel','reservation_status']]
plt.figure(figsize=(7,7))
sns.countplot(data =x, x = 'reservation_status', hue = 'hotel').set_title('Reservation status by hotel',fontsize=20)
# plt.show()
print('21] end')

# 22] get the most correlated data's
print('22] start')
corr_list=df[['lead_time','previous_cancellations','previous_bookings_not_canceled','booking_changes','days_in_waiting_list','adr','required_car_parking_spaces','total_of_special_requests','total_stay','total_guests']]
# corrdata=corr_list.corr()
# print(corrdata)
# plt.figure(figsize=(10,5))
# print(sns.heatmap(corrdata))
corrmat=corr_list.corr()
plt.subplots(figsize=(12,7))
sns.heatmap(corrmat,annot=True,fmt='.2f',annot_kws={'size': 10},vmax=.8,square=True)
plt.show()
print('22] end')
