RATE= ['Main Refinancing Rate', #EUR
 'Federal Funds Rate' , #USD
  'BOJ Policy Rate', #JPN
  'Cash Rate', #AUD
  'Official Cash Rate', #NZD
  'Official Bank Rate', #GBP
  'SNB Policy Rate' , #CHF
 ]

CPI = ['Core CPI Flash Estimate y/y', 'CPI Flash Estimate y/y', 'Italian Prelim CPI m/m', 'CPI m/m', 'CPI y/y', 'French Final CPI m/m', 'German Final CPI m/m', 'Core CPI y/y', 'Final CPI y/y', 'Final Core CPI y/y', 'CPI q/q', 'Core CPI m/m', 'Trimmed Mean CPI q/q', 'Tokyo Core CPI y/y', 'National Core CPI y/y', 'BOJ Core CPI y/y', 'French Prelim CPI m/m', 'Spanish Flash CPI y/y', 'Common CPI y/y', 'Median CPI y/y', 'Trimmed CPI y/y'] +\
     [ 'Core PCE Price Index m/m', "RICS House Price Balance"]+\
     ['RPI y/y', 'FPI m/m', ] + \
     ['GDT Price Index'] +\
     ["BRC Shop Price Index y/y",] +\
     ["ANZ Commodity Prices m/m", "Commodity Prices y/y",] + \
     ["German WPI m/m"]

PPI= ['IPPI m/m', 'PPI m/m', 'PPI y/y', 'Core PPI m/m', 'PPI Input m/m', 'PPI Output m/m', 'German PPI m/m', 'SPPI y/y', 'PPI q/q', 'PPI Input q/q', 'PPI Output q/q'] + \
     ['RMPI m/m'] 

GDP  = ['NIESR GDP Estimate', 'GDP q/y', 'Prelim GDP q/q', 'French Flash GDP q/q', 'Spanish Flash GDP q/q', 'GDP m/m', 'Advance GDP q/q', 'Advance GDP Price Index q/q', 'German Prelim GDP q/q', 'Italian Prelim GDP q/q', 'Flash GDP q/q', 'Prelim GDP Price Index y/y', 'German Final GDP q/q', 'Second Estimate GDP q/q', 'French Prelim GDP q/q', 'Prelim GDP Price Index q/q', 'GDP q/q', 'Final GDP q/q', 'Final GDP Price Index y/y', 'Revised GDP q/q', 'Final GDP Price Index q/q', 'Prelim Flash GDP q/q']

PMI = ['Manufacturing PMI', 'Non-Manufacturing PMI', 'Final Manufacturing PMI', 'Caixin Manufacturing PMI', 'Spanish Manufacturing PMI', 'Italian Manufacturing PMI', 'French Final Manufacturing PMI', 'German Final Manufacturing PMI', 'ISM Manufacturing PMI', 'Construction PMI', 'Caixin Services PMI', 'Spanish Services PMI', 'Italian Services PMI', 'French Final Services PMI', 'German Final Services PMI', 'Final Services PMI', 'ISM Services PMI', 'Retail PMI', 'Ivey PMI', 'Flash Manufacturing PMI', 'French Flash Manufacturing PMI', 'French Flash Services PMI', 'German Flash Manufacturing PMI', 'German Flash Services PMI', 'Flash Services PMI', 'Chicago PMI']


####### JOBS and EMPLOYMENT #######
EMPLOYMENT = ['Spanish Unemployment Change', 'German Unemployment Change', 'ADP Non-Farm Employment Change', 'Italian Monthly Unemployment Rate', 'Unemployment Rate', 'Employment Change', 'Non-Farm Employment Change', 'Spanish Unemployment Rate', 'Employment Cost Index q/q', 'Employment Change q/q', 'Italian Quarterly Unemployment Rate', 'Final Employment Change q/q', 'Flash Employment Change q/q']
JOB = ['JOLTS Job Openings', 'Challenger Job Cuts y/y', 'ANZ Job Advertisements m/m']
CLAIMS = ['Claimant Count Change', 'Unemployment Claims', ]


####### SENTIMENT #######
SENTIMENT= ['Economy Watchers Sentiment', 'Prelim UoM Consumer Sentiment', 'German ZEW Economic Sentiment', 'ZEW Economic Sentiment', 'Westpac Consumer Sentiment', 'Revised UoM Consumer Sentiment'] +\
     ['Sentix Investor Confidence', 'Consumer Confidence', 'NZIER Business Confidence', 'NAB Business Confidence', 'CB Consumer Confidence', 'GfK Consumer Confidence', 'NAB Quarterly Business Confidence', 'ANZ Business Confidence', 'Prelim ANZ Business Confidence'] +\
     ['German ifo Business Climate', 'Belgian NBB Business Climate', 'German GfK Consumer Climate', 'SECO Consumer Climate'] + \
     ['CBI Industrial Order Expectations'] +\
     ['KOF Economic Barometer'] +\
     ['Leading Indicators', 'CB Leading Index m/m', 'MI Leading Index m/m']+\
     ['UBS Economic Expectations'] +\
     ["NFIB Small Business Index"] +\
     [ "RCM/TIPP Economic Optimism"] +\
    ["UBS Consumption Indicator"]

SPENDING = ['Credit Card Spending y/y', 'Household Spending y/y', 'French Consumer Spending m/m', 'Personal Spending m/m', 'Capital Spending q/y']


####### Central Banks #######
ECB = ['ECB Monetary Policy Meeting Accounts', 'ECB Press Conference', 'ECB President Draghi Speaks', 'ECB Economic Bulletin', 'ECB Financial Stability Review', 'ECB President Lagarde Speaks']
FOMC = ['FOMC Member Fischer Speaks', 'FOMC Meeting Minutes', 'FOMC Member Bullard Speaks','FOMC Member Dudley Speaks', 'FOMC Member George Speaks','FOMC Member Rosengren Speaks', 'FOMC Member Mester Speaks','FOMC Member Powell Speaks', 'FOMC Member Brainard Speaks','FOMC Press Conference', 'FOMC Member Tarullo Speaks','FOMC Member Evans Speaks', 'FOMC Member Harker Speaks','FOMC Member Kashkari Speaks', 'FOMC Member Kaplan Speaks','FOMC Statement', 'FOMC Member Quarles Speaks','FOMC Member Williams Speaks', 'FOMC Member Bostic Speaks','FOMC Member Barkin Speaks', 'FOMC Member Daly Speaks','FOMC Member Clarida Speaks', 'FOMC Financial Stability Report','FOMC Member Bowman Speaks', 'FOMC Economic Projections','FOMC Member Waller Speaks', 'FOMC Member Collins Speaks','FOMC Member Jefferson Speaks', 'FOMC Member Cook Speaks','FOMC Member Barr Speaks', 'FOMC Member Logan Speaks','FOMC Member Goolsbee Speaks', 'FOMC Member Kugler Speaks','FOMC Member Musalem Speaks', 'FOMC Member Schmid Speaks','FOMC Member Hammack Speaks']
BOE = ['BOE Gov Carney Speaks', 'BOE Credit Conditions Survey', 'BOE Monetary Policy Report', 'BOE Inflation Letter', 'BOE Quarterly Bulletin', 'BOE Financial Stability Report', 'BOE Gov Bailey Speaks'] + \
        ['MPC Official Bank Rate Votes', 'MPC Asset Purchase Facility Votes', 'Asset Purchase Facility']
SNB = ['SNB Chairman Jordan Speaks', 'SNB Monetary Policy Assessment', 'SNB Quarterly Bulletin', 'SNB Financial Stability Report', 'SNB Press Conference', 'SNB Chairman Schlegel Speaks']
RBA = ['RBA Rate Statement', 'RBA Monetary Policy Statement', 'RBA Gov Stevens Speaks', 'RBA Assist Gov Edey Speaks', 'RBA Deputy Gov Debelle Speaks', 'RBA Deputy Gov Lowe Speaks', 'RBA Bulletin', 'RBA Assist Gov Kent Speaks', 'RBA Financial Stability Review', 'RBA Gov Lowe Speaks', 'RBA Annual Report', 'RBA Gov Bullock Speaks', 'RBA Assist Gov Ellis Speaks', 'RBA Press Conference', 'RBA Assist Gov Hunter Speaks', 'RBA Deputy Gov Hauser Speaks']

####### Political Events #######
BREXIT = ['Parliament Brexit Vote']

####### Trade #######
TRADE = ['Trade Balance', 'Goods Trade Balance', 'German Trade Balance', 'French Trade Balance', 'USD-Denominated Trade Balance', 'Italian Trade Balance', 'Overseas Trade Index q/q']
IMPORT = ['Import Prices m/m', 'Import Prices q/q', 'German Import Prices m/m']

####### Industrial #######
INDUSTRIAL = ['German Industrial Production m/m', 'French Industrial Production m/m', 'Industrial Production m/m', 'Italian Industrial Production m/m', 'Revised Industrial Production m/m', 'Industrial Production y/y', 'Prelim Industrial Production m/m']
SERVICES = [ "Index of Services 3m/3m","Tertiary Industry Activity m/m","All Industries Activity m/m",  "AIG Services Index",  "BusinessNZ Services Index", 'Tankan Non-Manufacturing Index']
MANUFACTURING = ['AIG Manufacturing Index', 'ISM Manufacturing Prices', 'Manufacturing Production m/m', 'Empire State Manufacturing Index', 'Manufacturing Sales m/m', 'BusinessNZ Manufacturing Index', 'Philly Fed Manufacturing Index', 'Richmond Manufacturing Index', 'Manufacturing Sales q/q', 'BSI Manufacturing Index', 'Tankan Manufacturing Index']
ORDERS= ['Factory Orders m/m', 'German Factory Orders m/m', 'Core Machinery Orders m/m', 'Prelim Machine Tool Orders y/y', 'Core Durable Goods Orders m/m', 'Durable Goods Orders m/m'] + \
        ['Private Capital Expenditure q/q'] +\
        ['Fixed Asset Investment ytd/y']
INVESTMENT =["Prelim Business Investment q/q", "Revised Business Investment q/q",]
PROFIT = ["Corporate Profits q/q", "Company Operating Profits q/q"]
INVENTORIES =["Business Inventories m/m",]
WAGE= ['Wage Price Index q/q'] + \
      ['Average Cash Earnings y/y', 'Average Hourly Earnings m/m', 'Average Earnings Index 3m/y'] + \
      ['Labor Market Conditions Index m/m', 'Labor Cost Index q/q', 'Prelim Unit Labor Costs q/q', 'Revised Unit Labor Costs q/q'] +\
     [  "French Prelim Private Payrolls q/q", "French Final Private Payrolls q/q"] +\
     ["Personal Income m/m",]


RETAIL = ['German Retail Sales m/m', 'Retail Sales m/m', 'Retail Sales y/y', 'BRC Retail Sales Monitor y/y', 'Core Retail Sales m/m', 'Italian Retail Sales m/m', 'Retail Sales q/q', 'Core Retail Sales q/q'] +\
        ['Final Wholesale Inventories m/m', 'Wholesale Sales m/m', 'Prelim Wholesale Inventories m/m'] + \
        [ "Wards Total Vehicle Sales", "CBI Realized Sales",  'New Motor Vehicle Sales m/m']

INFLATION = ['Prelim UoM Inflation Expectations', 'MI Inflation Gauge m/m', 'MI Inflation Expectations', 'Revised UoM Inflation Expectations', 'Inflation Expectations q/q', 'Consumer Inflation Expectations', 'Cleveland Fed Inflation Expectations']
HOUSE_PRICE = ['Halifax HPI m/m', 'NHPI m/m', 'Rightmove HPI m/m', 'ONS HPI y/y', 'HPI m/m', 'S&P/CS Composite-20 HPI y/y', 'Nationwide HPI m/m', 'HPI q/q', 'HPI y/y']

HOUSING = ['Building Approvals m/m', 'Building Permits m/m', 'Building Consents m/m', 'Building Permits'] + \
            ['Construction Spending m/m', 'AIG Construction Index', 'Construction Output m/m', 'Construction Work Done q/q'] + \
            ['Housing Starts', 'NAHB Housing Market Index', 'Housing Starts y/y', 'Housing Equity Withdrawal q/q'] +\
            [ 'HIA New Home Sales m/m', 'Existing Home Sales', 'New Home Sales', 'Pending Home Sales m/m', 'New Home Prices m/m']

LOAN = ['Consumer Credit m/m','Home Loans m/m', 'New Loans', 'Private Loans y/y',] +\
        [ "High Street Lending", "Net Lending to Individuals m/m","Bank Lending y/y", ] + \
        ["Mortgage Delinquencies", "Mortgage Approvals", ] + \
        [ "Private Sector Credit m/m",]

COMMODITY = ['Crude Oil Inventories', "Natural Gas Storage"]
OTHER_RATES = ['Libor Rate', '1-y Loan Prime Rate', '5-y Loan Prime Rate', 'Overnight Rate']

BOND = ['10-y Bond Auction', '30-y Bond Auction', 'French 10-y Bond Auction', 'German 10-y Bond Auction', 'German 30-y Bond Auction', 'Italian 10-y Bond Auction', 'Spanish 10-y Bond Auction']
BALANCE_OF_PAYMENT = ['Foreign Currency Reserves',"Long Term Refinancing Operation", "Public Sector Net Borrowing","French Gov Budget Balance","Federal Budget Balance",'Current Account', "Foreign Direct Investment ytd/y", "TIC Long-Term Purchases",  "Foreign Securities Purchases",]
MONEY = ['Monetary Base y/y','M4 Money Supply m/m', 'M2 Money Stock y/y', 'M2 Money Supply y/y', 'M3 Money Supply y/y']
PRODUCTIVITY = ['Prelim Nonfarm Productivity q/q', 'Revised Nonfarm Productivity q/q', 'Labor Productivity q/q'] +\
                ["Capacity Utilization Rate",]

TOURISM = ["Visitor Arrivals m/m",]

