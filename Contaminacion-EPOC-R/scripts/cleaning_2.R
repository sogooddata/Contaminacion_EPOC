# Script en el que se realiza la limpieza de los datos de contaminación 
# entre 2009 y 2015
library(data.table); library(lubridate)

load("raw_data/cont_2009_2015.RData")
# Se elige una magnitud de estudio
MAGN <- 8 #NO2
cont <- as.data.table(cont_2009_2015)[MAGNITUD == MAGN]

# Se definen las fechas y valores horarios del contaminante
FECHA <- ymd(paste(cont$ANO, "/", cont$MES, "/", cont$DIA))

MEDIDAS <- cont[, c(seq(9,55,2))]
MEDIDAS$H09 <- as.numeric(MEDIDAS$H09)
# Nuestros datos hospitalarios tienen granularidad diaria, así que los
# ponemos al mismo nivel
MEDIDAS <- rowMeans(MEDIDAS)

# Se hace la media para toda la ciudad al no tener información espacial directa
# de los hospitales
cont_n <- data.table(FECHA, cont$ESTACION, MEDIDAS)
cont_n[,NO2:=mean(MEDIDAS), by = list(FECHA)]

no2 <- unique(cont_n[,c(1,4)])
# save(no2, file="cleaned_data/no2.RData", compress = "xz")
