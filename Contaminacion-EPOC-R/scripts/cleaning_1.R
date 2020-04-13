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
# epoc_data <- epoc_data[grepl("491|492|496",`Diagnóstico Principal`)]
# epoc_data <- epoc_data[grepl("491|492|493|495|496",paste(`Diagnóstico Principal`, `Diagnóstico 2`, `Diagnóstico 3`,
#                                              `Diagnóstico 4`, `Diagnóstico 5`, `Diagnóstico 6`, `Diagnóstico 7`,
#                                              `Diagnóstico 8`, `Diagnóstico 9`, `Diagnóstico 10`))]

# save(epoc_data, file = "cleaned_data/epoc_data.RData", compress = "xz")

# Count es epoc_data[year(Fecha de ingreso) == c(2013,2014,2015),.N, by = list(Hospital)] después de ordenarlo de mayor a menor.
# par(mfrow=c(2,1))
# barplot(height = casos_totales_13a15$N, names = casos_totales_13a15$Hospital, las = 2,  cex.names = 0.75)
# barplot(height = count$N, names =count$`Hospital Recodificado`, las = 2, cex.names = 0.75)