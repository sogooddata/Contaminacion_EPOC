# SCRIPT IN WHICH EPOC SERIES IS ANALYZED AS A TIME SERIES
library(data.table); library(forecast); library(ggplot2); library(lubridate); library(plyr)

load("cleaned_data/epoc_data.RData")

# 
fechas <- data.table("Fecha de ingreso" = seq(ymd('2008-01-01'),ymd('2015-12-31'),by='day'))

# The number of total cases per day is calculated. As there is no patient repetition for each 
# hospitalization, this can be done easily with .N
epoc_casos <- epoc_data[order(`Fecha de ingreso`),.N, by = list(`Fecha de ingreso`)]
epoc_casos <- join(fechas, epoc_casos)
epoc_casos[is.na(epoc_casos)] = 0
colnames(epoc_casos)[2] <- "Número de casos"

# The time series definition. Two seasonal periods are considered: weekly and annualy. The annual
# one is trivial, the weekly one may not be so clear.
epoc_ts <- msts(epoc_casos$`Número de casos`, seasonal.periods = c(7, 365.25), start = c(2008, 1))
print(findfrequency(epoc_ts)) # This function serve as proof of weekly seasonality.

# Basic plots of the epoc series.
autoplot(epoc_ts) + labs(title = "Time series - EPOC", x = "Date", y = "Number of patients") + theme_bw()
autoplot(mstl(epoc_ts)) + labs(title = "Time series decomposition - EPOC", x = "Date") + theme_bw()

# For a more concise understanding of the series, it might be helpful its boxplots for different 
# time windows.
Sys.setlocale("LC_TIME", "C")
epoc_casos$weekday <- weekdays(epoc_casos$`Fecha de ingreso`)

ggplot(data = epoc_casos, aes(x = weekday, y = `Número de casos`, group = weekday)) +
  geom_boxplot(fill = "lightgrey") +
  scale_x_discrete(limits = c("Monday", "Tuesday", "Wednesday", "Thursday",
                              "Friday", "Saturday", "Sunday")) +
  labs(x = "Weekday", y = "Number of patients") +
  theme_bw() +
  theme(axis.text = element_text(size=12),
        text = element_text(size=14),
        axis.text.x = element_text(angle=45, hjust=1))

ggplot(data = epoc_casos, aes(x = month(`Fecha de ingreso`), y = `Número de casos`, group = month(`Fecha de ingreso`))) +
  geom_boxplot(fill = "lightgrey") +
  scale_x_discrete(limits = c("January", "February", "March", "April",
                              "May", "June", "July", "August",
                              "September", "October", "November", "December")) +
  labs(x = "Month", y = "Number of patients") +
  theme_bw() +
  theme(axis.text = element_text(size=12),
        text = element_text(size=14),
        axis.text.x = element_text(angle=45, hjust=1))

# Autocorrelation and Partial Autocorrelation plot
ggAcf(epoc_ts) + labs(title = "EPOC - Autocorrelation" ) + theme_bw()
ggPacf(epoc_ts) + labs(title = "EPOC - Partial Autocorrelation" ) + theme_bw()

# From this, it is clear the 365 days seasonality and the ausence of a 
# pronunciated trend. Lets see what happens if we just represent a 3-weeks period
ggAcf(epoc_ts, lag.max = 21) + labs(title = "EPOC - Autocorrelation" ) + theme_bw()


# Lastly, stationarity
library(tseries)
adf.test(epoc_ts)
