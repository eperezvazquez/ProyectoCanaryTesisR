# Machine Learning projects stocks.
About this data set, The Standard and Poor's 500, or S&P 500, is the most famous financial benchmark in the world.
This stock market index tracks the performance of 500 large companies in the United States on stock exchanges. As of December 31, 2020, more than $5.4 trillion was invested in assets tied to the performance of this index.
Because the index includes multiple classes of stock of some constituent companies—for example, Alphabet's Class A (GOOGL) and Class C (GOOG)—there are 505 stocks in the gauge.

The S&P 500 is a stock market index that tracks the largest 500 publicly traded U.S. companies. Investors have long used the S&P 500 as a benchmark for their investments as it tends to signal overall market health. The S&P 500 is a "free-floating index," meaning that it only considers the health and price of publicly traded shares; it does not consider government-owned or privately-owned shares. The index is a popular choice for long-term investors who wish to watch growth over the coming decades.
In our case, we used this stock market index to analyze the shares of the companies in the market and then focus on the semiconductor companies. Why did we choose the companies in the semiconductor technology industry? Because it has a high impact on the economy and specifically on technology such that they can cause a conflict between countries, as it later develops at work. Once the EDA is analyzed, we identify certain seasonality with daily patterns and variability. That is why we understood that the model that best fits is the Facebook prophet model.

Prophet es una herramienta de código abierto de Facebook utilizada para pronosticar datos de series de tiempo que ayuda a las empresas a comprender y posiblemente predecir el mercado. Se basa en un modelo aditivo descomponible donde las tendencias no lineales se ajustan a la estacionalidad, también tiene en cuenta los efectos de las vacaciones.
Once the model is selected, we observe that it has a daily seasonality, and so we apply the model where we identify which is the one that best fits with:

The r-squared is close to 1, so we can conclude that the fit is good. An r-squared value over 0.9 is impressive (and probably too good to be true, which tells me this data is most likely overfit). The value obtained is:
0.9738806280436687

MSE:
5758.239812750492
That's a considerable MSE value... and confirms my suspicion that this data is overfitting and won't likely hold up well into the future. Remember... for MSE, closer to zero is better.

And finally, the MAE result:
41.97861914441674

In summary, both in the application of Time Series at the level of the General Index with all the companies and the story of the three selected companies are AVGO Broadcom, MPWR Monolithic Power Systems Inc, LRCX Lam Research Corporation Lam Research, in both cases we find that time series seasonal manages to better predict the model with a confidence interval of 95% and weekly_seasonality=False and changepoint_prior_scale=0.9.
For the following cases, we will make predictions for each company using a multivariable model. For example, models MLP Multivariant (https://www.youtube.com/watch?v=87c9D_41GWg)


To see our project, PLEASE CHOOSE THIS URL TO SEE THE PROJECT: https://sp500-semiconductors.herokuapp.com/.





Note https://es.acervolima.com/analisis-de-series-de-tiempo-usando-facebook-prophet/
https://facebook.github.io/prophet/docs/trend_changepoints.html#automatic-changepoint-detection-in-prophet
Indicators: https://s-ai-f.github.io/Time-Series/time-series-forecasting.html
