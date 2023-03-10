## Motivation

Uncleared margin rules (UMR) apply to the exchange of margin collateralising uncleared OTC derivatives. From a collateral pledgor point-of-view, firms in scope of these rules may be required to/may opt to deliver collateral to a third party (custodian) that subsequently allocates said collateral to the firm's counterparties. For each given counterparty, dependent on the current trading exposure versus the counterparty, the firm has a required value (RQV) that they must meet. Operationally speaking, the firm is required to deliver securities from their internal inventory to their custodian giver account (or longbox), the custodian then allocates these securities from the giver account to the respective counterparty taker accounts. The value of collateral held on each taker account must equal at least the current counterparty RQV; if it is short the firm will either hold a deficit against the given counterparty or the custodian will offer utilisation of a cash credit line (at a fee) to cover the remainder of the required value.
  
At present, custodians often offer two methods of allocation: automatic or manual. Automatic means that securities pledged to the firm's longbox are allocated out by the custodian algorithmically under a given methodology. Often, these allocation methodologies are rigid in so far as they may not accurately reflect the business' changing internal cost of funding for their given inventory of securities. Manual means that the firm provides the custodian with an allocation file detailing the notional values of each security that should allocated to each counterparty's taker account. Naturally, creating an allocation file manually is laborious and near impossible to optimise with a large inventory/set of counterparties. Compounding this problem, counterparties have bespoke haircut schedules (which determine the value of collateral cover dependent on the nature of the given security) and concentration limit schedules (which impose limits on the volume of speficic types of securities alocated) that make compiling a manual allocation even more difficult.
  
The purpose of this project is to provide a method for generating manual allocations that both abide by the constraints imposed by each counterparty's haircut/concentration limit schedules and optimise expenditure in line with the firm's dynamic internal cost of securities funding. This project formulates the scenario as a linear programming (LP) problem and uses an open source solver.

## Description
optimiser_script.py contains an example script a user would run in order to formulate the daily UMR Allocation problem as an LP problem and subsequently find an optimal solution (if feasible). In brief, optimiser_script performs the following:  
  
- Imports live inventory/margin requirement data from the user's local environment.
- Imports live cost factor data. These are cost of funding coefficients determined by the user that reflect the cost of funding securities dependent on their ticker.
- Imports currency holiday data. Holidays prohibit movement of securities between internal inventory and custodian, this allows the program to ring-fence holiday collateral at the custodian.
- Imports priority securities list. These are manually listed by the user and contain securities that the user wishes to maximise usage of at the custodian, even if they result in a higher net funding cost (eg. if the user is lumbered with collateral they feel must be used at the given custodian).
- On an entity-by-entity basis, performs the following:
    - Requests user input for desired end-of-day longbox excess (the user may wish to leave a standard buffer to account for overnight security valuation changes).
    - Requests user input for linear programming tolerance levels (to account for small floating point errors).
    - Initialises an LP problem (cost minimisation).
    - Assigns an objective function to the problem which equals the cost funding a given allocation of securities (this is to be minimised).
    - Assigns constraints to the problem (available inventory, RQV cover, concentration limits), importing shelved data when necessary to calculate collateral haircuts and concentration limits.
    - Assigns elastic constraints to the problem related to priority securities.
    - Applies the LP solver to the problem:
        - If the problem is feasible, an allocation is returned that optimises funding costs (ie. finds the objective function's global minimum).
        - If the problem is infeasible (ie. insufficient collateral to cover all required values), the program will initialise a new LP problem (cover maximisation). This problem seeks a maximal collateral cover value (after haircut) given the current inventory. The program then returns a deficit report, a predicted net short and a report of any collateral the program was unable to allocate (due to ineligiblity/concentration limits). The user is then expected to source more collateral such that the initial cost minimisation problem is feasible.

optimiser_script.py utilises classes/methods stored in base_module.py. Listed below is a brief description of some of the main classes.

*Security_Profile*  
Object that contains static profile data for each security. This object is created and then shelved on a local database. The user can then request to read profile data from the entries of the database. Security profile data is used in when considering how a security will be valued respective to a counterparty's haircut schedule. It is also used to calculate eligibility/maximum capacity of a security respective to a counterparty's concentration limit schedule. I have chosen to shelve these objects rather than request for upload due to the data being static. Class methods allow for static data amendment.
  
*Security*  
Object containing current security balance, price, FX, priority status and cost factor data.

*Inventory*  
Collection of Security objects associated with given entity.

*Counterparty_Profile*  
Object that contains counterparty profile and haircut/concentration limit schedules. This object is created and then shelved on a local database. I have chosen to shelve these objects rather than request for upload due to the data being static. Class methods allow for static data amendment.

*Margin*  
Object containing current counterparty margin data, including current required value.

*Margin_Set*  
Collection of Margin objects associated with givn entity.

*Funding_Set*  
Object used to easily bind an entity's Inventory and Margin_Set.

*Model*  
Contains data/methods the user can employ to create LP problems and related objective functions/constraints. These are bound to Funding_Set objects.

## Prerequisites

This module requires the following Python modules:  
pandas (https://pandas.pydata.org/)  
NumPy (https://numpy.org/)  
PuLP (https://www.coin-or.org/PuLP/)  
shelve (included in standard Python library)