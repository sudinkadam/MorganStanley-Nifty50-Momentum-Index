# Submission by Sudin Kadam - sudinkadam@gmail.com
# Assumptions
# 1. The Risk Free Rate is assumed to be constant at 4% annual
# 2. Issuer Weight capping is not done since the bhacopy data source does not have the issuer data
# 3. Ongoing event changes are assumed to be included in the parent index 
# 4. The threshold delta on monthly change in volatility is assumed to be 0.13 to artificially trigger one momnth-end rebalancing due to high-volatility

import os
from bhavcopy import bhavcopy
import datetime
import pandas as pd
from pandas import *
import camelot
import numpy as np
from scipy.stats import zscore
from pandas.tseries.offsets import BDay, BMonthEnd


class Nift50IndexData:
    '''
    The Class has methods to parse the monthly Nifty 50 index composition PDF files present in the indexCompPdfPath folder defined below
    '''
    def __init__(self, projectPath):
        #IndexData related constants
        self.projectPath = projectPath
        self.indexCompPdfPath = self.projectPath + "indexData\indices_data{}\\NIFTY_50_{}.pdf"
        self.indexDataStartDate = "2023-01-01"
        self.indexDataEndDate = "2024-07-01"

        self.dfIndexCompDateRange = self.fetchNifty50CompositionDataForADateRange()

    def fetchNifty50CompositionDataForADateRange(self, ):
        ## DATA PREPARATION STEP1 : FETCHING THE MONTHWISE NIFT50 INDEX COMPOSITION AND THEIR WEIGHTS IN THE INDEX
        # Parse and read Month-Year wise Nifty50 Index Composition and MCap PDFs and Store it in a DataFrame
        # The monthwsie PDFs are manualy downloaded from https://www.niftyindices.com/reports/historical-data and 
        # saved in the indexData subfolder in {projectPath}
        
        indexCompPdfPath = projectPath + "indexData\indices_data{}\\NIFTY_50_{}.pdf"
        
        # Generate range of months and format as 'MonYYYY' (e.g., Aug2025)
        dates = pd.date_range(start=self.indexDataStartDate, end=self.indexDataEndDate, freq="MS")
        
        dfIndexCompList = []
        for d in dates:
            monthYear = d.strftime("%b%Y")
        
            # Extract tables from the PDFs
            tables = camelot.read_pdf(indexCompPdfPath.format(monthYear, monthYear), pages="all")
            
            #Each pdf has 2 pages on which 2 tables are present for the 50 index constituents. They are fetched in the following for loop
            dfIndexCompMonthYearList = []
            for i in range(2):
                # Convert tables to DataFrame
                df = tables[i].df
                
                df.columns = df.iloc[0]       # set first row as header
                df = df.drop(df.index[0])     # drop the first row
                df = df.reset_index(drop=True)
                df['MonthYear'] = monthYear
                df['Date'] = d
                dfIndexCompList.append(df)
            
            dfIndexCompRaw = concat(dfIndexCompList)
            colsToFilter = ['Symbol', 'Date', 'MonthYear', 'Close Price', 'Index Mcap\n(Rs. Crores)', 'Weightage\n(%)']
            dfIndexCompMonthYear = dfIndexCompRaw[colsToFilter]
            dfIndexCompMonthYearList.append(dfIndexCompMonthYear)
        
        colsToRename = {'Symbol':'Symbol', 'Close Price':'ClosePrice', 'Index Mcap\n(Rs. Crores)':'MCap', 'Weightage\n(%)': 'Weight'}
        dfIndexComp = concat(dfIndexCompList).rename(columns=colsToRename)

        return dfIndexComp

    def getLatestInfty50CompositionData(self, runDate):
        #Fetching the Nifty50 Index Universe For The Latest Month End Before the rundate

        dfIndexCompSorted = self.dfIndexCompDateRange.set_index('Date').sort_index()
        dfIndexCompSortedTruncated = dfIndexCompSorted.truncate(after=runDate)
        latestIndexComposition = dfIndexCompSortedTruncated[dfIndexCompSortedTruncated.index==dfIndexCompSortedTruncated.index.max()]
        universe = latestIndexComposition.Symbol.to_list()

        return universe, latestIndexComposition



class EquityPriceData():
    def __init__(self, projectPath):
        #IndexData related constants
        self.equityDataPath = projectPath + "Data\\"
        fileName = "equities.csv"

        if os.path.exists(self.equityDataPath+fileName):
            dfEquitiesRaw = read_csv(self.equityDataPath+fileName, parse_dates=['TIMESTAMP'])
        else:
            self.fetchEquitiesPriceDataUsingbhavcopyPackage(self.equityDataPath+fileName)
            dfEquitiesRaw = read_csv(self.equityDataPath+fileName, parse_dates=['TIMESTAMP'])
            
        dfEquitiesRaw = dfEquitiesRaw[dfEquitiesRaw.SERIES=='EQ']
        
        colsToFilterEq = ['TIMESTAMP', 'SYMBOL', 'CLOSE']
        self.dfEquitiesRaw = dfEquitiesRaw[dfEquitiesRaw['SERIES']=='EQ'][colsToFilterEq]

    def fetchEquityPriceDataFfill(self, ):
        '''
        This function processes the daily price series data for Nifty stocks for further use
        '''
        dfEquitiesRaw = self.dfEquitiesRaw

        # Get the full business day index
        all_days = pd.date_range(dfEquitiesRaw["TIMESTAMP"].min(), dfEquitiesRaw["TIMESTAMP"].max(), freq="B")
        
        # Forward fill for each stock
        dfEquitiesRawFfill = (
                                dfEquitiesRaw.set_index("TIMESTAMP")
                                             .groupby("SYMBOL")
                                             .apply(lambda x: x.reindex(all_days).ffill())
                                             .drop("SYMBOL", axis=1, errors="ignore")
                                             .reset_index()
                                             .rename(columns={"level_1": "TIMESTAMP"})
                            )
        
        dfEquitiesRawFfill = dfEquitiesRawFfill.set_index('TIMESTAMP').sort_index()

        return dfEquitiesRawFfill

    def month_end_prices(self, daily_prices: pd.DataFrame) -> pd.DataFrame:
        """
        Resample daily adjusted close prices to month-end close.
        """
        dfMonthEndPrices = daily_prices.groupby("SYMBOL")["CLOSE"].resample("ME").last().reset_index()
        dfMonthEndPrices = dfMonthEndPrices.set_index('TIMESTAMP').sort_index()
        return dfMonthEndPrices
    
    def week_end_prices(self, daily_prices: pd.DataFrame) -> pd.DataFrame:
        """
        Resample daily adjusted close prices to week-end (Fri) close.
        """
        # 'W-FRI' week ending on Friday; if market holiday, resample('W-FRI').last() uses last obs in that week.
        dfWeeklyPrices = daily_prices.groupby("SYMBOL")["CLOSE"].resample("W-FRI").last().reset_index()
        dfWeeklyPrices = dfWeeklyPrices.set_index('TIMESTAMP').sort_index()
        return dfWeeklyPrices

    data_storage = r"C:\Users\sud12\Documents\ISS Coding\Data\\"

    def fetchEquitiesPriceDataUsingbhavcopyPackage(self, data_storage, start_date = datetime.date(2020, 1, 1), end_date = datetime.date(2025, 7, 31)):
        '''
        The function uses the bhacopy package to fetch nifty price series data between given start and end dates
        '''
        # Define working directory, where files would be saved
        os.chdir(data_storage)
        
        # Define wait time in seconds to avoid getting blocked
        wait_time = [1, 2]
        
        nse = bhavcopy("equities", start_date, end_date, data_storage, wait_time)
        nse.get_data()

class MomentumIndexConstructor():
    '''
    This class does the Nifty50 momentum index initial construction, semi-annual rebalancing and the conditional monthly high vol rebalancing according to
    the approach mentioned in the MSCI paper.
    It uses the equities daily price data, Nifty 50 index composition and weights data
    '''
    def __init__(self, startDate, endDate, projectPath = r"C:\Users\sud12\Documents\ISS Coding\\"):
        self.projectPath = r"C:\Users\sud12\Documents\ISS Coding\\"
        self.indexCompositionDict = {}

    def computeMomentumZScores(self, dfEquitiesRawFfill, dfWeeklyPrices, universe, highVolRebalance=False):
        #Section 2.2.0 Computation of Price Momentum

        riskFreeReturns12M = 0.04                                         #To be changed
        riskFreeReturns6M = riskFreeReturns12M/2.0
        maxDate = runDate - pd.DateOffset(months=1)
        
        dfEquitiesRawFfill6M = dfEquitiesRawFfill.truncate(before=runDate - pd.DateOffset(months=7), after=maxDate).reset_index()
        priceMomPivot6M = pd.pivot_table(dfEquitiesRawFfill6M, index = 'TIMESTAMP', columns = 'SYMBOL', aggfunc='mean')
        
        dfEquitiesRawFfill12M = dfEquitiesRawFfill.truncate(before=runDate - pd.DateOffset(months=13), after=maxDate).reset_index()
        priceMomPivot12M = pd.pivot_table(dfEquitiesRawFfill12M, index = 'TIMESTAMP', columns = 'SYMBOL', aggfunc='mean')
        
        priceMom6M = priceMomPivot6M.iloc[-1] / priceMomPivot6M.iloc[0] - 1 - riskFreeReturns6M
        priceMom6M = priceMom6M.reset_index().drop(columns='level_0').set_index('SYMBOL').reindex(universe).rename(columns={0:'priceMom6M'})
        
        priceMom12M = priceMomPivot12M.iloc[-1] / priceMomPivot12M.iloc[0] - 1 - riskFreeReturns12M
        priceMom12M = priceMom12M.reset_index().drop(columns='level_0').set_index('SYMBOL').reindex(universe).rename(columns={0:'priceMom12M'})
        
        priceMom6M12M = pd.concat([priceMom6M, priceMom12M], axis=1)

        #Section 2.2.1 Computation of Risk Adjusted Momemntum
        dfWeeklyPricesLast3Y = dfWeeklyPrices.truncate(before=runDate-pd.DateOffset(months=36), after=runDate)
        priceWeeklyPivot36M = pd.pivot_table(dfWeeklyPricesLast3Y, index = 'TIMESTAMP', columns = 'SYMBOL', aggfunc='mean')
        
        returnsWeekly36M = priceWeeklyPivot36M.pct_change()
        stdevReturnsWeekly36M = returnsWeekly36M.std()*np.sqrt(52)
        stdevReturnsWeekly36M = stdevReturnsWeekly36M.reset_index().drop(columns='level_0').set_index('SYMBOL').reindex(universe).rename(columns={0:'stdevReturnsWeekly36M'})
        
        riskAdjPriceMom6M = (priceMom6M12M['priceMom6M']/stdevReturnsWeekly36M['stdevReturnsWeekly36M'])
        riskAdjPriceMom6M.name = 'riskAdjPriceMom6M'
        riskAdjPriceMom12M = (priceMom6M12M['priceMom12M']/stdevReturnsWeekly36M['stdevReturnsWeekly36M'])
        riskAdjPriceMom12M.name = 'riskAdjPriceMom12M'
        
        riskAdjPriceMom6M12M = pd.concat([riskAdjPriceMom6M, riskAdjPriceMom12M], axis=1)

        #Section 2.2.2 Computation of The Momentum Score
        riskAdjPriceMom6M12MZScore = riskAdjPriceMom6M12M.apply(zscore)
        if not highVolRebalance:
            momCombinedScoreC = riskAdjPriceMom6M12MZScore.mean(axis=1)
        else:
            momCombinedScoreC = riskAdjPriceMom6M12MZScore['riskAdjPriceMom6M']
        momCombinedScoreC.name = 'momCombinedScore'
        
        standardizedMomZscore = pd.Series(zscore(momCombinedScoreC), index=momCombinedScoreC.index)
        standardizedMomZscore.name = 'standardizedMomZscore'
        
        winsStandardizedMomZscore = standardizedMomZscore.clip(upper=3, lower=-3)
        momWinsZScore = pd.Series(np.where(winsStandardizedMomZscore>0, 1+winsStandardizedMomZscore, 1.0/(1-winsStandardizedMomZscore)), index = winsStandardizedMomZscore.index, name='momWinsZScore')
        
        unWinsMomZScore = pd.Series(np.where(standardizedMomZscore>0, 1+standardizedMomZscore, 1.0/(1-standardizedMomZscore)), index = winsStandardizedMomZscore.index, name='unWinsMomZScore')

        return momWinsZScore, unWinsMomZScore

    def roundOffRules(self, numSecPrevStep):
        if numSecPrevStep<100:
            nearestRounding = 10
        elif numSecPrevStep>=100 and numSecPrevStep<300:
            nearestRounding = 25
        else:
            nearestRounding = 50
        return nearestRounding
    
    def round_to_nearest(self, x, base):
        return base * round(x / base)
        
    def fixedNumberOfSecuritiesInitialConstruction(self, df, parentIndMCapCoverage=30):
        '''
        Computes the number of secutiries at the initial Momntum Index consruction using method in Appendix1
        Inputs:  
            - Parent index constituents dataframe
            - The mcap coverage number (%) in the parent mcap
        Output: number of securities (int)
        '''
        minNumSecFor30pcMCapCoverage = len(df[df['cumMCap%']<=parentIndMCapCoverage])
        nearestRounding = self.roundOffRules(minNumSecFor30pcMCapCoverage)
        roundedMinNumSecForGivenParentIndMCapCoverage = self.round_to_nearest(minNumSecFor30pcMCapCoverage, nearestRounding)
        parentIndexSecuritiesCount = len(df)
        print('minNumSecFor30pcMCapCoverage = ', minNumSecFor30pcMCapCoverage)
        print('roundedMinNumSecFor30pcMCapCoverage = ', roundedMinNumSecForGivenParentIndMCapCoverage)
    
        if parentIndexSecuritiesCount<=25:
            numSec = parentIndexSecuritiesCount
        elif minNumSecFor30pcMCapCoverage<=25 or minNumSecFor30pcMCapCoverage<=0.1*parentIndexSecuritiesCount:
            if minNumSecFor30pcMCapCoverage<=25:
                numSec = 25
            else:
                numSec = minNumSecFor30pcMCapCoverage<=0.1*parentIndexSecuritiesCount
        elif roundedMinNumSecForGivenParentIndMCapCoverage>=0.4*parentIndexSecuritiesCount:
            while roundedMinNumSecForGivenParentIndMCapCoverage>0.4*parentIndexSecuritiesCount:
                roundedMinNumSecForGivenParentIndMCapCoverage = roundedMinNumSecForGivenParentIndMCapCoverage-1
                print('roundedMinNumSecForGivenParentIndMCapCoverage = ', roundedMinNumSecForGivenParentIndMCapCoverage)
                if df[df['Rank']==roundedMinNumSecForGivenParentIndMCapCoverage]['cumMCap%'].values[0]<20:
                    nearestRounding = self.roundOffRules(roundedMinNumSecForGivenParentIndMCapCoverage+1)
                    numSec = self.round_to_nearest(roundedMinNumSecForGivenParentIndMCapCoverage+1, nearestRounding)
                    return numSec
            nearestRounding = self.roundOffRules(roundedMinNumSecForGivenParentIndMCapCoverage)
            numSec = self.round_to_nearest(roundedMinNumSecForGivenParentIndMCapCoverage, nearestRounding)
        else:
            numSec = roundedMinNumSecForGivenParentIndMCapCoverage
        
        return numSec
    
    def fixedNumberOfSecuritiesSemiAnnualRebalancing(self, df, prevNumSec, parentIndMCapCoverage=30):
        '''
        Computes the number of secutiries at rebalancing using method in Appendix2
        Inputs:  
            Parent index constituents dataframe
            The number of securities at the time of previous construction
        Output: number of securities (int)
        '''
        # print('prevNumSec =', prevNumSec)
        parentIndexSecuritiesCount = len(df)
        print('New parentIndexSecuritiesCount =', parentIndexSecuritiesCount)
        
        if prevNumSec>parentIndexSecuritiesCount:
            numSec = self.fixedNumberOfSecuritiesInitialConstruction(df)
        elif parentIndexSecuritiesCount<=25:
            numSec = parentIndexSecuritiesCount
        elif prevNumSec<25:
            numSec = self.fixedNumberOfSecuritiesInitialConstruction(df)
        elif df[df['Rank']==prevNumSec]['cumMCap%'].values[0]<10:
            numSec = self.fixedNumberOfSecuritiesInitialConstruction(df)
        else:
            numSec = prevNumSec
    
        return numSec

    def weightingScheme(self, latestNifty50Universe, latestIndexComposition, momWinsZScore, finalIndexConstituents):
        # Section 2.4: WEIGHTING SCHEME for the securities in the momentum index
        latestIndexCompositionWt = latestIndexComposition[['Symbol', 'Weight']].set_index('Symbol').reindex(latestNifty50Universe)
        latestIndexCompositionWt['Weight'] = latestIndexCompositionWt['Weight'].astype(float)
        
        momCombinedScoreWithLatestIndexWt = pd.concat([momWinsZScore, latestIndexCompositionWt], axis=1)
        momCombinedScoreWithLatestIndexWt = momCombinedScoreWithLatestIndexWt.reindex(finalIndexConstituents)
        
        momCombinedScoreWithLatestIndexWt["Momentum Weight"] = momCombinedScoreWithLatestIndexWt["momWinsZScore"]*momCombinedScoreWithLatestIndexWt["Weight"]
        momCombinedScoreWithLatestIndexWt["Std Momentum Weight"] = momCombinedScoreWithLatestIndexWt["Momentum Weight"]/momCombinedScoreWithLatestIndexWt["Momentum Weight"].sum()

        return momCombinedScoreWithLatestIndexWt
        
    def initialConstructionAndRebalancing(self, runDate, dfEquitiesRawFfill, dfWeeklyPrices, latestNifty50Universe, latestNifty50Composition, rebalancing=False, prevMomIndex=None, highVolRebalance=False):
        '''
        The function does initial momentum index constructino and the subsequent rebalancing
        Inputs:
            - runDate - date of rebalancing/initial construction [datetime]
            - dfEquitiesRawFfill - daily price data              [Dataframe]
            - dfWeeklyPrices - weekly price data                 [Dataframe]
            - latestNifty50Universe - latest nifty50 univese     [list ]
            - latestNifty50Composition - latest nifty50 composition and weights    [Dataframe]
            - rebalancing: Whether rebalancing run or the initial construction run [Boolean]
        Output: The momentum index with constituents and the securities weight composition
        '''
        momWinsZScore, unWinsMomZScore = self.computeMomentumZScores(dfEquitiesRawFfill, dfWeeklyPrices, latestNifty50Universe, highVolRebalance)

        #Getting The Fixed Number of Securities at Initial Construction / Semi Annual Rebalancing
        latestIndexCompositionWt = latestNifty50Composition[['Symbol', 'Weight', 'MCap']].set_index('Symbol').reindex(latestNifty50Universe)
        latestIndexCompositionWt['Weight'] = latestIndexCompositionWt['Weight'].astype(float)
        latestIndexCompositionWt['MCap'] = latestIndexCompositionWt['MCap'].astype(float)
        
        unWinsMomZScoreWithLatestIndexWt = pd.concat([unWinsMomZScore, latestIndexCompositionWt], axis=1)
        
        sortedmUnWinsMomZScoreWithLatestIndexWt = unWinsMomZScoreWithLatestIndexWt.sort_values(by=["unWinsMomZScore", "Weight"], ascending=[False, False])
        sortedmUnWinsMomZScoreWithLatestIndexWt["Rank"] = range(1, len(sortedmUnWinsMomZScoreWithLatestIndexWt) + 1)
        # sortedmUnWinsMomZScoreWithLatestIndexWt.loc[sortedmUnWinsMomZScoreWithLatestIndexWt["Rank"]<22, 'MCap'] = 0   #Used for testing
        sortedmUnWinsMomZScoreWithLatestIndexWt['cumMCap%'] = sortedmUnWinsMomZScoreWithLatestIndexWt['MCap'].cumsum()/sortedmUnWinsMomZScoreWithLatestIndexWt['MCap'].sum()*100

        dfMomZScore = sortedmUnWinsMomZScoreWithLatestIndexWt.copy()
        if not rebalancing:
            numSec = self.fixedNumberOfSecuritiesInitialConstruction(dfMomZScore)
        else:
            numSec = self.fixedNumberOfSecuritiesSemiAnnualRebalancing(dfMomZScore, len(prevMomIndex))
        print('numSec = ', numSec)
        finalIndexConstituents = dfMomZScore[dfMomZScore['Rank']<=numSec].index

        momCombinedScoreWithLatestIndexWt = self.weightingScheme(latestNifty50Universe, latestNifty50Composition, momWinsZScore, finalIndexConstituents)
        print('finalIndex and Weights =\n', momCombinedScoreWithLatestIndexWt[momCombinedScoreWithLatestIndexWt.index.isin(finalIndexConstituents)])
        return momCombinedScoreWithLatestIndexWt


    def dailyReturnVolLast3M(self, runDate, dailyEquitiesPrices, latestMomIndex):
        '''
        Computes the voltility of the momntum index based on last 3 months dail returns of the index constituents
        Inputs: 
            - dailyEquitiesPrices: Daily price series of stocks [Dataframe]
            - latestMomIndex: Latest Nifty50 index composition [Dataframe]
        Output:
            - Index volatility [float]
        '''
        dailyEquitiesPricesLast3M = dailyEquitiesPrices.truncate(before=runDate - pd.DateOffset(months=3)-BDay(), after=runDate).reset_index()
        pricePivot3M = pd.pivot_table(dailyEquitiesPricesLast3M, index = 'TIMESTAMP', columns = 'SYMBOL', aggfunc='mean')
    
        dailyReturnsLast3M = pricePivot3M.pct_change()
        dailyReturnsLast3M.columns = dailyReturnsLast3M.columns.droplevel(0)
        dailyReturnsLast3MRI = dailyReturnsLast3M.reindex(columns=latestMomIndex.index).T
    
        dailyMomIndexReturnsLast3M = dailyReturnsLast3MRI.mul(latestMomIndex['Std Momentum Weight'], axis=0).sum()
        momIndex3MVol = dailyMomIndexReturnsLast3M.std()*np.sqrt(252)
        return momIndex3MVol


if __name__ == '__main__':

    projectPath = r"C:\Users\sud12\Documents\ISS Coding\\"
    startDate = "2023-12-29"
    endDate = "2024-07-01"
    
    #Instantiation of class that fetches the nift50 Index composition data from the PDFs fetched from online sources and the associated methods
    NID = Nift50IndexData(projectPath)
    
    #Instantiation of class that fetches the daily stock prices for all the nifty stocks bewteen start and end dates and the associated methods
    EPD = EquityPriceData(projectPath)
    dfEquitiesRawFfill = EPD.fetchEquityPriceDataFfill()
    dfWeeklyPrices = EPD.week_end_prices(dfEquitiesRawFfill)
    dfMonthEndPrices = EPD.month_end_prices(dfEquitiesRawFfill)
    
    #Instantiation of class that fetches the daily stock prices for all the nifty stocks bewteen start and end dates
    MIC = MomentumIndexConstructor(startDate, endDate)
    
    # Business days from 1st Jan 2024 to 1st Jan 2025
    bdates = pd.bdate_range(start=startDate, end=endDate)
    regularRebalanceDates = pd.bdate_range(start=startDate, end=endDate, freq=6 * BMonthEnd())
    regularRebalanceDates = regularRebalanceDates[1:]
    
    monthEndDates = pd.bdate_range(start=startDate, end=endDate, freq=1 * BMonthEnd())
    monthlyVolComputeDates = [dt-9*BDay() for dt in monthEndDates]
    
    momentumIndex = DataFrame()
    thresholdMomIndex3MVol = 0.13
    previousMonthMomIndex3MVol = -1
    
    for runDate in bdates[:]:
        latestNifty50Universe, latestNifty50Composition = NID.getLatestInfty50CompositionData(runDate)
        if momentumIndex.empty:
            print('----- Doing Initial Momentum Index Construction ----')
            print('runDate = ', runDate)
            momentumIndex = MIC.initialConstructionAndRebalancing(runDate, dfEquitiesRawFfill, dfWeeklyPrices, latestNifty50Universe, latestNifty50Composition)
            momentumIndex.to_csv(projectPath+"output\\momentumIndex_Initial_"+runDate.strftime("%Y-%m-%d")+".csv")
            continue
    
        if runDate in regularRebalanceDates:
            print('----- Doing Momentum Index Rebalancing -----')
            print('runDate = ', runDate)
            momentumIndex = MIC.initialConstructionAndRebalancing(runDate, dfEquitiesRawFfill, dfWeeklyPrices, latestNifty50Universe, latestNifty50Composition, rebalancing=True, prevMomIndex=momentumIndex)
            momentumIndex.to_csv(projectPath+"output\\momentumIndex_SemiAnnRebalance_"+runDate.strftime("%Y-%m-%d")+".csv")
            continue
    
        if runDate in monthEndDates:
            print('----- Computing Last 3M Momentum Index Volatility -----')
            print('runDate = ', runDate)
            momIndex3MVol = MIC.dailyReturnVolLast3M(runDate, dfEquitiesRawFfill, momentumIndex)
            print('momIndex3MVol = ', momIndex3MVol)
            if previousMonthMomIndex3MVol!=-1:
                delta = previousMonthMomIndex3MVol/momIndex3MVol - 1
                print('delta =', delta)
                previousMonthMomIndex3MVol = momIndex3MVol
                if momIndex3MVol>thresholdMomIndex3MVol:
                    print('----- Doing Momentum Index Rebalancing Due To High Vol -----')
                    print('momIndex3MVol = ', momIndex3MVol)
                    print('thresholdMomIndex3MVol = ', thresholdMomIndex3MVol)
                    momentumIndex = MIC.initialConstructionAndRebalancing(runDate, dfEquitiesRawFfill, dfWeeklyPrices, latestNifty50Universe, latestNifty50Composition, rebalancing=True, prevMomIndex=momentumIndex, highVolRebalance=True)
                    momentumIndex.to_csv(projectPath+"output\\momentumIndex_HighVolRebalance_"+runDate.strftime("%Y-%m-%d")+".csv")
            else:
                previousMonthMomIndex3MVol = momIndex3MVol
