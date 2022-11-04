from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


from enum import Enum
import time
import json 

app = FastAPI()
WRONG_FORMAT_EXCEPTION = HTTPException(status_code = 400, detail = "Zählerstand hat falsches Format")
PATH_TO_CSV = "./data/GAS.csv"
PLOT_PATH = "./data/plots/"


class Reading(BaseModel):
    reading: str

class PlotTypes(Enum):
    ABSOLUTE = "ABSOLUTE"
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"


def check_plot_availability(plotType: PlotTypes):
    with open(PLOT_PATH+"existing_plots.json", "r") as f:
        existing_plots = json.loads(f.read())
    
    plot_metadata = existing_plots.get(plotType.name, None)
    if(plot_metadata is None):
        return False
    
    loc_datetime = time.localtime()
    if(loc_datetime.tm_year != plot_metadata["year"]): # recreate the plot if the it has not been done during the last 10 minutes
        return False
    if(loc_datetime.tm_mon != plot_metadata["month"]):
        return False
    if(loc_datetime.tm_mday != plot_metadata["day"]):
        return False
    if(loc_datetime.tm_hour != plot_metadata["hour"]):
        return False
    if(loc_datetime.tm_min - 10 > plot_metadata["minute"]): 
        return False
    return True

def update_plot_availability(plotType: PlotTypes):
    with open(PLOT_PATH+"existing_plots.json", "r") as f:
        existing_plots = json.loads(f.read())
    
    loc_datetime = time.localtime()
    existing_plots[plotType.name] = {   "year":loc_datetime.tm_year, 
                                        "month":loc_datetime.tm_mon, 
                                        "day":loc_datetime.tm_mday, 
                                        "hour":loc_datetime.tm_hour,
                                        "minute":loc_datetime.tm_min}

    with open(PLOT_PATH+"existing_plots.json", "w") as f:
        f.write(json.dumps(existing_plots))
    
def write_into_csv(number: float):
    loc_datetime = time.localtime()
    loc_date = "{}.{}.{}".format(loc_datetime.tm_mday, loc_datetime.tm_mon, loc_datetime.tm_year)
    loc_time = "{}:{}:{}".format(loc_datetime.tm_hour, loc_datetime.tm_min, loc_datetime.tm_sec)
    line = "{};{};{}\n".format(loc_date, number, loc_time)

    df = pd.read_csv(PATH_TO_CSV, sep = ";", index_col=False)
    
    if(number <= df["Gas"].max()):
        raise HTTPException(status_code = 400, detail = "Zählerstand zu gering")
    
    new_line = {
        "Date":loc_date,
        "Gas":number, 
        "Time":loc_time
    }
    df = pd.concat([df, pd.DataFrame([new_line])], ignore_index=True)
    df.to_csv(PATH_TO_CSV, sep = ";", index = False)

def create_absolute_plot():
    current_type = PlotTypes.ABSOLUTE

    plot_up_to_date = check_plot_availability(current_type)
    if(plot_up_to_date):
        return 

    df = pd.read_csv(PATH_TO_CSV, sep = ";", index_col=False)
    df['Date'] =  pd.to_datetime(df['Date'], format = "%d.%m.%Y")

    fig, ax = plt.subplots(1,1, figsize = (15,15))

    ax.plot(df["Date"], df["Gas"])
    ax.scatter(df["Date"], df["Gas"])

    ax.set_xlabel("Tag", size = 15)
    ax.set_ylabel("Zählerstand", size = 15)

    horizontal = 9950
    while horizontal <= df["Gas"].values[-1]:
        ax.axhline(horizontal, linestyle = ":", color = "black")
        horizontal += 25
    fig.savefig(PLOT_PATH+current_type.value +".png")
    update_plot_availability(current_type)

def create_daily_plot():
    current_type = PlotTypes.DAILY

    plot_up_to_date = check_plot_availability(current_type)
    if(plot_up_to_date):
        return 

    df = pd.read_csv(PATH_TO_CSV, sep = ";", index_col=False)
    df['Date'] =  pd.to_datetime(df['Date'], format = "%d.%m.%Y")

    daylist = []
    avg_consumption = []

    dates = df["Date"].values
    gasses = df["Gas"].values

    for i in range(df.shape[0]-1):
        curr = i+1
        daylist.append(curr)
        dayDelta = (dates[curr] - dates[i])/np.timedelta64(1, 'D')
        consumption = (gasses[curr] - gasses[i])/dayDelta
        avg_consumption.append(consumption)

    fig, ax = plt.subplots(1,1, figsize = (15,15))
    ax.plot(df["Date"].values[1:], avg_consumption)
    ax.scatter(df["Date"].values[1:], avg_consumption)

    ax.set_xlabel("Tag", size = 15)
    ax.set_ylabel("Durchschnittlicher täglicher Gasverbrauch(m^3)", size = 15)

    for elem in range(1,8):
        ax.axhline(elem, linestyle = ":", color = "black")

    fig.savefig(PLOT_PATH+current_type.value +".png")
    update_plot_availability(current_type)

def create_weekly_plot():
    current_type = PlotTypes.WEEKLY

    plot_up_to_date = check_plot_availability(current_type)
    if(plot_up_to_date):
        return 

    df = pd.read_csv(PATH_TO_CSV, sep = ";", index_col=False)
    df['Date'] =  pd.to_datetime(df['Date'], format = "%d.%m.%Y")

    daylist = []
    avg_consumption = []

    dates = df["Date"].values
    gasses = df["Gas"].values

    for i in range(df.shape[0]-1):
        curr = i+1
        daylist.append(curr)
        dayDelta = (dates[curr] - dates[i])/np.timedelta64(1, 'D')
        consumption = (gasses[curr] - gasses[i])/dayDelta
        avg_consumption.append(consumption)


    daylist = df['Date'].values[1:]
    gaslist = avg_consumption

    first_day = df['Date'].values[0] + np.timedelta64(1, 'D')
    current_monday = first_day

    factor = 0
    week_averages = []
    kws = []
    first_kw = 29

    while current_monday < df['Date'].values[-1]:
        current_monday = first_day + np.timedelta64(7*factor, 'D')
        kws.append("KW-{}".format(first_kw+factor))
        factor += 1
        current_weekday = current_monday
        weeks_values = []
        while current_weekday < current_monday + np.timedelta64(7, 'D'):
            if(current_weekday in daylist):
                index = np.where(daylist == current_weekday)
                #index = daylist.where(current_weekday)
                weeks_values.append(gaslist[index[0][0]])
            current_weekday +=  np.timedelta64(1, 'D')
        if len(weeks_values) == 0:
            week_averages.append(None)
        else:
            week_averages.append(sum(weeks_values)*7/len(weeks_values))

    kws = kws[:-1]
    week_averages = week_averages[:-1]

    for i, elem in enumerate(week_averages):
        if elem is not None:
            continue
        week_averages[i] = week_averages[i+1]

    fig, ax = plt.subplots(1,1, figsize = (15,15))
    ax.bar(kws, week_averages)

    ax.set_xlabel("Kalenderwoche", size = 15)
    ax.set_ylabel("Geschätzter Verbrauch der Woche(m^3)", size = 15)

    for elem in range(0,50, 5):
        ax.axhline(elem, linestyle = ":", color = "black")
    

    fig.savefig(PLOT_PATH+current_type.value +".png")
    update_plot_availability(current_type)


@app.get("/")
async def root():
    with open("./public/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)


@app.post("/log")
async def makeLog(reading: Reading):
    reading_string = reading.reading.replace(",",".")
    if(len(reading_string) != 9):
        raise WRONG_FORMAT_EXCEPTION
    try:
        reading_numerical = float(reading_string)
    except:
        raise WRONG_FORMAT_EXCEPTION
    if(reading_numerical > 99999.999 or reading_numerical <= 0):
        raise WRONG_FORMAT_EXCEPTION

    write_into_csv(reading_numerical)
    return {"detail":"success"}

@app.get("/plots/absolute")
async def get_plot_absolute():
    create_absolute_plot()
    return FileResponse(PLOT_PATH+PlotTypes.ABSOLUTE.value +".png")

@app.get("/plots/daily")
async def get_plot_daily():
    create_daily_plot()
    return FileResponse(PLOT_PATH+PlotTypes.DAILY.value +".png")

@app.get("/plots/weekly")
async def get_plot_weekly():
    create_weekly_plot()
    return FileResponse(PLOT_PATH+PlotTypes.WEEKLY.value +".png")



