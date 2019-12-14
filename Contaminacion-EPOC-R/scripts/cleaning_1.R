# SCRIPT EN EL QUE SE HACE UNA PRIMERA CAPA DE LIMPIEZA A LOS DATOS DE ENFERMEDADES RESPIRATORIAS.
# SIMPLEMENTE SE AJUSTAN CORRECTAMENTE LOS FORMATOS (FECHAS, DATOS GEOGRÁFICOS) Y SE FILTRAN AQUELLOS
# DIAGNÓSTICOS PRINCIPALES QUE TIENEN QUE VER CON EL EPOC (BRONQUITIS (491), ENFISEMA (492) Y EPOC (493)).
# SE DESCARTAN ASMA, RINITIS, Y OTRAS MUCHAS ENFERMEDADES QUE SIN DUDA PUEDEN TENER RELACIÓN CON LA CONTAMINACIÓN.

library(data.table); library(lubridate)

epoc_data <- fread("raw_data/CMBD_6_20181217-135856.csv")

# En primer lugar se adopta un formato legible de fechas
epoc_data[, c("Fecha de nacimiento", "Fecha de ingreso", "Fecha de alta", "Fecha de Intervención") := 
            list(dmy(`Fecha de nacimiento`), dmy(`Fecha de ingreso`), dmy(`Fecha de alta`), dmy(`Fecha de Intervención`))]

# Solo se trabaja con datos de la Comunidad de Madrid, por lo que ciertas columnas no son necesarias. Además
# la columna Fecha coincide con el año de la Fecha de alta.
epoc_data[, c("Año" ,"Provincia", "Comunidad Autónoma") := NULL]
epoc_data <- epoc_data[grepl("491|492|496",`Diagnóstico Principal`)]

save(epoc_data, file = "cleaned_data/epoc_data.RData", compress = "xz")

