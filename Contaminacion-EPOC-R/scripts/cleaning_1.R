# SCRIPT EN EL QUE SE HACE UNA PRIMERA CAPA DE LIMPIEZA A LOS DATOS DE ENFERMEDADES RESPIRATORIAS.
# SIMPLEMENTE SE AJUSTAN CORRECTAMENTE LOS FORMATOS: FECHAS, DATOS GEOGRÁFICOS

library(data.table); library(lubridate); library(plyr)

epoc_data <- fread("raw_data/CMBD_6_20181217-135856.csv")

# En primer lugar se adopta un formato legible de fechas
epoc_data[, c("Fecha de nacimiento", "Fecha de ingreso", "Fecha de alta", "Fecha de Intervención") := 
            list(dmy(`Fecha de nacimiento`), dmy(`Fecha de ingreso`), dmy(`Fecha de alta`), dmy(`Fecha de Intervención`))]

# Solo se trabaja con datos de la Comunidad de Madrid, por lo que ciertas columnas no son necesarias. Además
# la columna Fecha coincide con el año de la Fecha de alta.
epoc_data[, c("Año" ,"Provincia", "Comunidad Autónoma") := NULL]

save(epoc_data, file = "cleaned_data/epoc_data.RData", compress = "xz")

