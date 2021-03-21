url <- "https://raw.githubusercontent.com/selva86/datasets/master/ozone.csv"

input_data <- read.csv(url)

head(input_data)

boxplot(x = input_data$pressure_height)

boxplot.stats(x = input_data$pressure_height)

outlier_values <- boxplot.stats(x = input_data$pressure_height)$out

boxplot(ozone_reading ~ Month, data = input_data)
boxplot(data = input_data, ozone_reading ~ Day_of_week)

boxplot(ozone_reading ~ pressure_height, data = input_data)


boxplot(ozone_reading ~ cut(x = pressure_height, breaks = pretty(input_data$pressure_height)), data = input_data)

mod <- lm(formula = ozone_reading ~ . , data = input_data)

cooksd <- cooks.distance(mod)

cooksd[1:10]

plot(cooksd)
abline(h = 4*mean(cooksd, na.rm = T), col='red')
text(pos=4,x =1:length(cooksd), y = cooksd, labels = ifelse(test = cooksd>4*mean(cooksd, na.rm = T),yes = names(cooksd), no = ""))

dim(input_data)

outlier_obs <- as.numeric(names(cooksd)[cooksd>4*mean(cooksd, na.rm = T)])

outlier_obs

input_data[outlier_obs,]

print(installed.packages()[,c(1,3)])

car::outlierTest(model = mod)
# Suggestion is that row 130 is the most extreme


