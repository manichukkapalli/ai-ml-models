CREATE TABLE [Raw_Retail_Sales] (
	Id integer NOT NULL,
	ProductCategory varchar(255) NOT NULL,
	MonthlyNominalGDPIndexinMillion float NOT NULL,
	MonthlyRealGDPIndexinMillion float NOT NULL,
	CPI float NOT NULL,
	unemploymentrate float NOT NULL,
	CommercialBankInterestRateonCreditCardPlans float NOT NULL,
	FinanceRateonPersonalLoansatCommercialBanks24MonthLoan float NOT NULL,
	Earningsorwagesindollarsperhour float NOT NULL,
	CottonMonthlyPriceUScentsperPoundlbs float NOT NULL,
	Changein float NOT NULL,
	Averageuplandplantedmillionacres float NOT NULL,
	yieldperharvestedacre integer NOT NULL,
	Millusein480lbnetwerightinmillionbales float NOT NULL,
	Exports float NOT NULL,
	SeaLevelPressavg float NOT NULL,
	Visibilityavg float NOT NULL,
	Windavg float NOT NULL,
	Precipsum float NOT NULL,
	Event integer NOT NULL,
	FederalHoliday integer NOT NULL,
  CONSTRAINT [PK_RAW_RETAIL_SALES] PRIMARY KEY CLUSTERED
  (
  [Id] ASC
  ) WITH (IGNORE_DUP_KEY = OFF)

)
GO
CREATE TABLE [Prediction_Retail_Sales] (
	Prediction_Id integer NOT NULL,
	SalesInThousandDollars varchar(255) NOT NULL
)
GO

ALTER TABLE [Prediction_Retail_Sales] WITH CHECK ADD CONSTRAINT [Prediction_Retail_Sales_fk0] FOREIGN KEY ([Prediction_Id]) REFERENCES [Raw_Retail_Sales]([Id])
ON UPDATE CASCADE
GO
ALTER TABLE [Prediction_Retail_Sales] CHECK CONSTRAINT [Prediction_Retail_Sales_fk0]
GO
