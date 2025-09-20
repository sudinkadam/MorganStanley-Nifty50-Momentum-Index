About:
- This project is an adaptation of the Morgan Stanley's Momentum Index construction methodology for US markets based on the white paper published in Aug 2021 to indian markets. 
- It applies the methodology to Nifty50 which is a free-float market-cap weighted index for 50 largest and most liquid Indian stocks. The resultant index is referred to as Nifty50 Momentum Index. 
- The white paper is present in this repository as MSCI_Momentum_Indexes_Methodology_Aug2021.pdf file

- Nifty50 Index Data: The Nifty50 Momentum Index composition and constituent weights are obtained from the monthwise PDFs available on https://www.niftyindices.com/reports/historical-data . These PDF files are manually downloaded for period from Jan 2023 till Aug 2025 and saved in the accompanying folder at \indexData. The PDF files are parsed in the code to  fetch the monthwise constituents and their weights

- Price Data: The suggested bhavcopy Python library was used to fetch the equity data of Nifty listed companies. I have fetched the data for a periods from 2021-01-01 to 2024-07-05. The code by default fetches the price data from 1st Jan 2020 to 31st July 2025. But unfortunately, I was getting error in fetching data post July 2024. Also, I think my IP has got blocked due to excessive usage due to which I was unable to fetch data before Jan 2021. That is why my Nifty50 Momentum Index construction is over ~6 month period before July 2024.

Configuration:

- Set the following variables in the main function
	- projectPath = The path where the un-zipped folder shared on the mail is saved
	- startDate = Date from which momentum index is to be created 	[optional change]
	- endDate = Date upto which rebalancing is to be done		[optional change]

- The index constituents PDFs for period before 1st Jan 2023 need to be downloaded and saved in the \indexData folder [optional step]
- By default equities price data is fetched for period from 1st Jan 2025. Modify the function arguments in fetchEquitiesPriceDataUsingbhavcopyPackage to fetch it for different daterange					      [Default step for the 1st run]

------------------------------------------------------------
Output:
- The resultant Nifty50 Momentum Index csv files are stored in \output subfolder. The name of the CSV is momentumIndex_{RebalanceType}_{DateofRealance}.csv
- The CSV is indexed by stock symbols and has the following columns:
	- momWinsZScore		: The momentum z-score mentioned in section 2.2.2 in the paper
	- Weight		: The weight of the stock in the relevant Nifty50 index composition
	- Momentum Weight	: The momentum weight without standardization as mentioned in section 2.4
	- Std Momentum Weight	: The standardized momentum weight as mentioned in section 2.4. This is the weight of the stock in the Nifty 50 Momentum Index
- The first Nifty50 Momentum Index construction is done in Dec 2023
- The semiannual rebalancing is done in June 2024

------------------------------------------------------------
Assumptions:
- It is assumed that the 3-Month risk free rate that used in the computation of the price momentum remains constant at 4%. This is assumed because I was unable to find historical series of these returns in due time

- The value of threshold on monthly change in volatility delta is assumed to be 0.13 . This is due to 
	1. Artificially trigger one high-vol monthly rebalance in March 2024
	2. Inability to find the 95th percentile value of delta for a reference index in due time 
	3. It is assumed that only one stock per issue is present. Hence issuer Weight capping is not done
	4. Ongoing event changes are assumed to be included in the parent index due to inability to find the event changes data in due time
------------------------------------------------------------
