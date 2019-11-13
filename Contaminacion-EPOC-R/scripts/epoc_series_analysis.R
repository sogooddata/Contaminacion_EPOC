# SCRIPT EN EL QUE SE ANALIZA LA SERIE DE ENFERMEDADES EPOCS DESDE UN PUNTO DE VISTA DE SERIE TEMPORAL

library(data.table); library(forecast); library(ggplot2); library(lubridate); library(plyr)

load("cleaned_data/epoc_data.RData")

fechas <- data.table("Fecha de ingreso" = seq(ymd('2008-01-01'),ymd('2015-12-31'),by='day'))
epoc_casos <- epoc_data[order(`Fecha de ingreso`),.N, by = list(`Fecha de ingreso`)]
epoc_casos <- join(fechas, epoc_casos)
epoc_casos[is.na(epoc_casos)] = 0
colnames(epoc_casos)[2] <- "Número de casos"
epoc_ts <- msts(epoc_casos$`Número de casos`, seasonal.periods = c(7, 365.25), start = c(2008, 1))

autoplot(epoc_ts) + theme_bw()
autoplot(mstl(epoc_ts)) + theme_bw()
