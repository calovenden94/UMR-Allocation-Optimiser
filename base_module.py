import numpy as np
import pandas as pd
import pulp

# Class Definitions - Security (Static)

class Security_Profile:
    '''
    Object containing security profile data - static and shelved.
    '''

    def __init__(self, profile_data):
        
        self.ISIN = profile_data.ISIN
        self.currency = profile_data.currency
        self.ticker = profile_data.ticker
        self.minimum_increment_amount = profile_data.minimum_increment_amount
        self.maturity_date = profile_data.maturity_date


# Class Definitions - Security (Live)

class Security:
    '''
    Object containing current security balance, price, FX, priority status and cost factor data.
    '''

    def __init__(self, balance_data, cost_factor_data, priority_data, holiday_data, FX_data, price_data, security_database):

        self.entity = balance_data.entity
        self.ISIN = balance_data.ISIN
        self.currency = balance_data.currency
        self.ticker = security_database[self.ISIN].ticker
        self.cost_factor = lookup(self.ticker, cost_factor_data, 'ticker', 'cost_factor')
     
        try:
            self.priority_status = lookup(self.ISIN, priority_data, 'ISIN', 'priority_status')
        except:
            self.priority_status = 'No'

        if self.entity in holiday_data:
            self.holiday_status = 'Yes'
        else:
            self.holiday_status = 'No'

        self.USD_FX_rate = lookup(self.currency, FX_data, 'currency', 'USD_FX_rate')
        self.closing_price = lookup(self.ISIN, price_data, 'ISIN', 'closing_price')
        
        self.depot_notional = balance_data.depot_notional
        self.custodian_notional = balance_data.custodian_notional
        
        if self.holiday_status == 'Yes':
            self.available_notional = self.custodian_notional
        else:
            self.available_notional = self.depot_notional + self.custodian_notional

        self.available_MV = self.available_notional * self.closing_price
        self.available_MV_USD = self.available_MV * self.USD_FX_rate

    def Return_Haircut(self, counterparty_database, counterparty_ID):
        '''
        Method that returns haircut applied to given security when allocated to given counterparty.
        Counterparty haircut schedules are retrieved from shelve; database key = counterparty_ID.
        '''

        haircut = 1

        # Check for cross-currency haircut.

        if [x.rate for x in counterparty_database[counterparty_ID].haircut_schedule if x.haircut_type == 'xccy'] != []:
            haircut = haircut * [x.rate for x in counterparty_database[counterparty_ID].haircut_schedule if x.haircut_type == 'xccy'][0]

        # Compound with currency haircut.

        haircut = haircut * [x.rate for x in counterparty_database[counterparty_ID].haircut_schedule if x.currency == self.currency][0]

        return haircut

class Inventory:
    '''
    Collection of Security objects associated with given entity.
    '''

    def __init__(self, entity, security_list):
        
        self.entity = entity
        self.securities = security_list
        self.size = len(self.securities)


# Class Definitions - Counterparty (Static)

class Counterparty_Profile:
    '''
    Object containing counterparty profile and haircut/concentration limit schedules - static and shelved.
    '''

    def __init__(self, counterparty_data):

        # Base counterparty data required for initialisation.
        self.entity = counterparty_data.entity
        self.counterparty_ID = counterparty_data.counterparty_ID
        self.margin_currency = counterparty_data.margin_currency
        self.haircut_schedule = []
        self.concentration_limit_schedule = []

class Haircut_Rule():
    '''
    Object containing data for a single haircut rule.
    '''

    def __init__(self, haircut_data):

        self.counterparty_ID = haircut_data.counterparty_ID
        self.rate = haircut_data.rate
        self.haircut_type = haircut_data.haircut_type
        self.currency = haircut_data.currency

    # Decorator to provide data validation for haircut type.
    @property
    def haircut_type(self):
        return self._haircut_type
    
    @haircut_type.setter
    def haircut_type(self, value):
        if not value in ['ccy', 'xccy']:
            raise Exception('Type must be ccy or xccy.')
        self._haircut_type = value

class Concentration_Limit_Rule():
    '''
    Object containing data for a single concentration limit rule.
    '''

    def __init__(self, concentration_limit_data):

        self.counterparty_ID = concentration_limit_data.counterparty_ID
        self.limit_type = concentration_limit_data.limit_type
        self.limit_currency = concentration_limit_data.limit_currency
        self.limit_percentage = concentration_limit_data.limit_percentage
        self.limit_absolute_value = concentration_limit_data.limit_absolute_value
        self.security_currency = concentration_limit_data.security_currency
        self.security_nature = concentration_limit_data.security_nature
        self.security_maturity = concentration_limit_data.security_maturity

    'Decorator to provide data validation for concentration limit type.' 
    @property
    def limit_type(self):
        return self._limit_type
    
    @limit_type.setter
    def limit_type(self, value):
        if not value in ['Percentage', 'Absolute_Value']:
            raise Exception('Type must be Percentage or Absolute_Value.')
        self._limit_type = value


# Class Definitions - Counterparty (Live)

class Margin:
    '''
    Object containing current counterparty margin data.
    '''

    def __init__(self, margin_data, FX_data):

        self.entity = margin_data.entity
        self.counterparty_ID = margin_data.counterparty_ID
        self.RQV = margin_data.RQV
        self.margin_currency = margin_data.margin_currency
        self.USD_FX_rate = lookup(self.margin_currency, FX_data, 'currency', 'USD_FX_rate')
        self.RQV_USD = self.RQV * self.USD_FX_rate

    def Retrieve_Concentration_Limit_Schedule(self, counterparty_database):
        '''
        Method that retrieves concentration limit schedule from shelve and assigns it to live Margin object; database key = counterparty_ID.
        '''

        self.concentration_limit_schedule = counterparty_database[self.counterparty_ID].concentration_limit_schedule

class Margin_Set:
    '''
    Collection of Margin objects with RQVs (counterparties and associated longbox) associated with given entity.
    '''

    def __init__(self, entity, margin_list):
    
        self.entity = entity
        self.margins = margin_list
        self.size = len(self.margins)

    def Add_Longbox(self, desired_longbox_excess):

        self.desired_longbox_excess = desired_longbox_excess


# Class Definitions - Funding_Set/Model

class Funding_Set:
    '''
    Object that packages entity's Inventory and Margin_Set together.
    Created outside of related Model objects in order to generate haircut/weight matrices.
    '''

    def __init__(self, entity, inventory, margin_set):

        self.entity = entity
        self.inventory = inventory
        self.margin_set = margin_set

    def Generate_Haircut_Matrix(self, counterparty_database):
        '''
        Method that generates haircut matrix for given Funding_Set.
        i,j-th entry is haircut applied to i-th security upon allocation to j-th counterparty (by indices of Inventory/Margin_Set).
        '''

        matrix = np.zeros((self.inventory.size, self.margin_set.size))
        
        for i in range(self.inventory.size):
            for j in range(self.margin_set.size):
                matrix[i][j] = self.inventory.securities[i].Return_Haircut(counterparty_database, self.margin_set.margins[j].counterparty_ID)

        self.haircut_matrix = matrix

    def Generate_Weight_Matrices(self):
        '''
        Method that generates the following matrices for given Funding_Set:
            - USD_MV_conversion_vector - vector of weights equal to closing_price * USD_FX_rate for each security
            - cost_weight_vector - vector of weights equal to closing_price * USD_FX_rate * cost_factor for each security
            - allocation_weight_matrix - matrix of weights equal to closing_price * USD*FX_rate * haircut for each security/counterparty pair
            - allocation_weight_transpose - transpose of allocation_weight_matrix for ease of indexing
        '''

        self.USD_MV_conversion_vector = np.array([(x.closing_price * x.USD_FX_rate) for x in self.inventory.securities])
        self.cost_weight_vector = np.array([(x.closing_price * x.USD_FX_rate * x.cost_factor) for x in self.inventory.securities])

        self.allocation_weight_matrix = np.matmul(np.diag(self.USD_MV_conversion_vector), self.haircut_matrix)
        self.allocation_weight_transpose = np.transpose(self.allocation_weight_matrix)

class Model():
    '''
    Object containing entity's Inventory and Margin_Set and attached LP problem.
    '''

    def __init__(self, name, sense, Funding_Set):

        self.inventory = Funding_Set.inventory
        self.margin_set = Funding_Set.margin_set
        self.entity = Funding_Set.entity
        self.problem = pulp.LpProblem(self.entity + '_' + name, sense=sense)

        # Model inherits haircut/weight matrices from Funding_Set.

        self.haircut_matrix = Funding_Set.haircut_matrix
        self.USD_MV_conversion_vector = Funding_Set.USD_MV_conversion_vector
        self.cost_weight_vector = Funding_Set.cost_weight_vector
        self.allocation_weight_matrix = Funding_Set.allocation_weight_matrix
        self.allocation_weight_transpose = Funding_Set.allocation_weight_transpose

        # Model requests manual tolerance input on a model-by-model basis. This is to account for floating point multiplication inaccuracies and minimum increment amounts.
        # Priority penalty parameter is entered. This allows the user to decide the extent to which the LP will penalise the objective for leaving priority securities unused.

        self.RQV_cover_tolerance = \
            1 + (int(input('\n' + 'Please enter ' + self.entity + '_' + name + ' RQV cover tolerance (in 10,000ths of a percent). Recommended figure is 5-25.' + '\n')) * 0.000001)
        self.ISIN_usage_tolerance = \
            1 - (int(input('Please enter ' + self.entity + '_' + name + ' ISIN usage tolerance (in 10,000ths of a percent). Recommended figure is 5-25.' + '\n')) * 0.000001)
        self.conc_limit_tolerance = \
            1 - (int(input('Please enter ' + self.entity + '_' + name + ' concentration limit tolerance (in 10,000ths of a percent). Recommended figure is 5-25.' + '\n')) * 0.000001)
        self.priority_penalty = \
            float(input('Please enter ' + self.entity + '_' + name + ' remaining priority securities penalty (in x times cost value of remaining MV). Recommended figure is 2-5.' + '\n'))

    # Decision variable creation.

    def Create_LpVariable_Allocation_Matrices(self):
        '''
        Method that creates allocation decision variable matrix/transpose for given Inventory and Margin_Set. 
        '''
    
        self.allocation_notional_matrix = \
            pulp.LpVariable.matrix('x', indices=([x.ISIN for x in self.inventory.securities], [x.counterparty_ID for x in self.margin_set.margins]), lowBound=0)
        self.allocation_notional_transpose = np.transpose(self.allocation_notional_matrix)

    def Create_LpVariable_Longbox_Matrices(self):
        '''
        Method that creates longbox decision variable matrix/transpose for given Inventory. 
        '''

        self.longbox_notional_matrix = pulp.LpVariable.matrix('x', indices=([x.ISIN for x in self.inventory.securities], ['Longbox']), lowBound=0)
        self.longbox_notional_transpose = np.transpose(self.longbox_notional_matrix)

    # Objective function assignment.

    def Assign_LpAffineExpression_Cost_Minimisation(self):
        '''
        Method that assigns cost minimisation objective function to given problem.
        '''

        objective_sum = 0

        for j in range(self.margin_set.size):
            objective_sum += pulp.lpDot(self.cost_weight_vector, self.allocation_notional_transpose[j])

        objective_sum += pulp.lpDot(self.cost_weight_vector, self.longbox_notional_transpose)
            
        self.problem += objective_sum

    def Assign_LpAffineExpression_Cover_Maximisation(self):
        '''
        Method that assigns cover maximisation objective function to given problem.
        '''

        objective_sum = 0

        for j in range(self.margin_set.size):
            objective_sum += pulp.lpDot(self.allocation_weight_transpose[j], self.allocation_notional_transpose[j])

        self.problem += objective_sum

    # Constraint assignment.

    def Assign_LpConstraint_RQV_Cover(self):
        '''
        Method that assigns RQV cover constraints to the given problem.
        '''
        
        for j in range(self.margin_set.size):
            self.problem += pulp.lpDot(self.allocation_weight_transpose[j], self.allocation_notional_transpose[j]) - (self.margin_set.margins[j].RQV_USD * self.RQV_cover_tolerance) >= 0

    def Assign_LpConstraint_RQV_Cap(self):
        '''
        Method that assigns RQV cap constraints to the given problem.
        '''

        for j in range(self.margin_set.size):
            self.problem += pulp.lpDot(self.allocation_weight_transpose[j], self.allocation_notional_transpose[j]) <= (self.margin_set.margins[j].RQV_USD * self.RQV_cover_tolerance)

    def Assign_LpConstraint_Longbox_Cover(self):
        '''
        Method that assigns a longbox cover constraint to the given problem.
        '''

        self.problem += pulp.lpDot(self.USD_MV_conversion_vector, self.longbox_notional_transpose) - (self.margin_set.desired_longbox_excess * self.RQV_cover_tolerance) >= 0

    def Assign_LpConstraint_ISIN_Usage(self):
        '''
        Method that assigns maximal ISIN usage constraints to the given problem.
        '''

        if hasattr(self, 'longbox_notional_matrix'):
            for i in range(self.inventory.size):
                self.problem += pulp.lpSum(self.allocation_notional_matrix[i]) + self.longbox_notional_matrix[i] <= (self.inventory.securities[i].available_notional * self.ISIN_usage_tolerance)
        else:
            for i in range(self.inventory.size):
                self.problem += pulp.lpSum(self.allocation_notional_matrix[i]) <= (self.inventory.securities[i].available_notional * self.ISIN_usage_tolerance)     

    def Assign_LpConstraint_Priority_Securities(self):
        '''
        Method that assigns priority usage constraints to the given problem. These constraints are created and elasticised. The problem is then extended to include them.
        Penalty term for elastic constraint is chosen such that impact of remaining priority inventory is its cost multiplied by the priority penalty chosen by the user.
        Method also assigns constraints that ensure priority securities are not assigned to longbox.
        '''

        priority_status_index = [i for i, x in enumerate(self.inventory.securities) if x.priority_status == 'Yes']

        priority_securities_constraint_list = [None] * len(priority_status_index)
        priority_ESP_list = [None] * len(priority_status_index)

        for i in range(len(priority_status_index)):

            priority_securities_constraint_list[i] = \
                pulp.lpSum(self.allocation_notional_matrix[priority_status_index[i]]) == self.inventory.securities[priority_status_index[i]].available_notional
            
            priority_ESP_list[i] = \
                priority_securities_constraint_list[i].makeElasticSubProblem(penalty = \
                    self.priority_penalty * self.cost_weight_vector[priority_status_index[i]], proportionFreeBoundList = [1 - self.ISIN_usage_tolerance, 0])
            
            priority_ESP_list[i].name = 'p_ESP' + str(i)

            self.problem.extend(priority_ESP_list[i])

        for i in range(len(priority_status_index)):
            self.problem += self.longbox_notional_transpose[priority_status_index[i]] == 0

    def Assign_LpConstraint_Holiday_Securities(self):
        '''
        Method that assigns holiday security usage constraints to the given problem. These constraints are created and elasticised. The problem is then extended to include them.
        Penalty term for elastic constraint is such that impact of allocating holiday inventory to longbox is double its cost.
        '''

        holiday_status_index = [i for i, x in enumerate(self.inventory.securities) if x.holiday_status == 'Yes']

        holiday_securities_constraint_list = [None] * len(holiday_status_index)
        holiday_ESP_list = [None] * len(holiday_status_index)

        for i in range(len(holiday_status_index)):

            holiday_securities_constraint_list[i] = self.longbox_notional_transpose[holiday_status_index[i]] == 0

            holiday_ESP_list[i] = \
                holiday_securities_constraint_list[i].makeElasticSubProblem(penalty = 2 * self.cost_weight_vector[holiday_status_index[i]], proportionFreeBoundList = [0, 0])
            
            holiday_ESP_list[i].name = 'h_ESP' + str(i)

            self.problem.extend(holiday_ESP_list[i])

        # LpConstraint objects are created and assigned - ensuring holiday securities are all utilised.
        # Buffer of 5 * ISIN_usage_tolerance used to ensure feasibility. Holiday ISINs are snapped up to full usage after allocation.

        for i in range(len(holiday_status_index)):
            self.problem += pulp.lpSum(self.allocation_notional_matrix[holiday_status_index[i]]) >= \
                self.inventory.securities[holiday_status_index[i]].available_notional * (1 - (5 * (1 - self.ISIN_usage_tolerance)))
            
    def Assign_LpConstraint_Concentration_Limits(self, FX_data):
        '''
        Method that assigns concentration limit constraints to problem.
        '''

        for j in range(self.margin_set.size):
            for k in range(len(self.margin_set.margins[j].concentration_limit_schedule)):

                limit_type = self.margin_set.margins[j].concentration_limit_schedule[k].limit_type
                limit_currency = self.margin_set.margins[j].concentration_limit_schedule[k].limit_currency
                limit_percentage = self.margin_set.margins[j].concentration_limit_schedule[k].limit_percentage
                limit_absolute_value = self.margin_set.margins[j].concentration_limit_schedule[k].limit_absolute_value
                security_currency = self.margin_set.margins[j].concentration_limit_schedule[k].security_currency

                security_currency_index = [i for i, x in enumerate(self.inventory.securities) if x.currency == security_currency]

                if security_currency_index != []:

                    # Compile a subset of decision variables and their associated weights.

                    notional_compiler = []
                    weight_compiler = []

                    for i in security_currency_index:

                        notional_compiler.append(self.allocation_notional_matrix[i][j])
                        weight_compiler.append(self.allocation_weight_matrix[i][j])

                    if limit_type == 'Percentage':

                        self.problem += pulp.lpDot(notional_compiler, weight_compiler) <= limit_percentage * self.margin_set.margins[j].RQV_USD * self.conc_limit_tolerance

                    elif limit_type == 'Absolute_Value':
                        
                        self.problem += pulp.lpDot(notional_compiler, weight_compiler) <= limit_absolute_value * lookup(limit_currency, FX_data, 'currency', 'USD_FX_rate') * self.conc_limit_tolerance

    # Solver application.

    def Apply_Solver(self):
        '''
        Method that applies GLPK_CMD solver to given problem.
        '''

        self.solver = pulp.GLPK_CMD()
        self.problem.solve(self.solver)

    # Result matrix population.

    def Populate_Allocation_Result_Matrix(self):
        '''
        Method that populates allocation result matrix with solution to given problem.
        '''

        self.allocation_result_matrix = np.zeros((self.inventory.size, self.margin_set.size))

        for i in range(self.inventory.size):
            for j in range(self.margin_set.size):
                self.allocation_result_matrix[i][j] = self.allocation_notional_matrix[i][j].varValue

    def Populate_Longbox_Result_Matrix(self):
        '''
        Method that populates longbox result matrix with solution to given problem.
        '''

        self.longbox_result_matrix = np.zeros((self.inventory.size, 1))

        for i in range(self.inventory.size):
            self.longbox_result_matrix[i] = self.longbox_notional_matrix[i][0].varValue

    # Result matrix amendment.

    def Security_Usage_Amendment(self):
        '''
        Method that determines whether a given security should be fully utilised (to avoid leaving small amounts of unused securities).
        Once the given securities are determined, the remaining notional amount is added to the longbox allocation.
        '''

        for i in range(self.inventory.size):
            if (self.inventory.securities[i].percentage_utilisation > self.ISIN_usage_tolerance) or \
                ((self.inventory.securities[i].percentage_utilisation >= 0.99) & (self.inventory.securities[i].remaining_MV_USD <= 1000)) or \
                (self.inventory.securities[i].holiday_status == 'Yes'):
                
                self.longbox_result_matrix[i] += self.inventory.securities[i].remaining_notional  

    def Minimum_Increment_Amendment(self, result_matrix, security_database):
        '''
        Method that rounds result matrix entries down according to given security's minimum increment amount.
        '''

        matrix_rows, matrix_columns = np.shape(result_matrix)

        for i in range(matrix_rows):
            for j in range(matrix_columns):

                increment = security_database[self.inventory.securities[i].ISIN].minimum_increment_amount

                result_matrix[i][j] = np.floor(result_matrix[i][j] / increment) * increment    

    # Result matrix entry aggregation.

    def Security_Usage_Calculation(self, allocation_result_matrix, longbox_result_matrix=None):
        '''
        Method that populates remaining notional/MV fields for given inventory after allocation has been generated.
        Argument for longbox_result_matrix is optional given that model may not have created a matrix of this kind.
        '''

        if np.shape(longbox_result_matrix) == ():
            longbox_result_matrix = np.zeros((self.inventory.size, 1))

        for i in range(self.inventory.size):
            self.inventory.securities[i].notional_used = allocation_result_matrix[i].sum() + longbox_result_matrix[i].sum()
            self.inventory.securities[i].remaining_notional = self.inventory.securities[i].available_notional - self.inventory.securities[i].notional_used
            self.inventory.securities[i].remaining_MV_USD = \
                self.inventory.securities[i].remaining_notional * self.inventory.securities[i].USD_FX_rate * self.inventory.securities[i].closing_price
            self.inventory.securities[i].percentage_utilisation = \
                (self.inventory.securities[i].available_MV_USD - self.inventory.securities[i].remaining_MV_USD)/self.inventory.securities[i].available_MV_USD
            self.inventory.securities[i].transfer_value = self.inventory.securities[i].notional_used - self.inventory.securities[i].custodian_notional
            
    def Margin_Cover_Calculation(self):
        '''
        Method that calculates margin cover for given margin set after allocation has been generated.
        A post-HC cover matrix is calculated by element-wise mutliplication of the allocation result matrix and the allocation weight matrix.
        '''

        cover_matrix = self.allocation_result_matrix * self.allocation_weight_matrix

        for j in range(self.margin_set.size):
            self.margin_set.margins[j].post_haircut_cover_USD = cover_matrix[:, j].sum()
            self.margin_set.margins[j].excess_USD = self.margin_set.margins[j].post_haircut_cover_USD - self.margin_set.margins[j].RQV_USD   

    # Constraint satisfaction.

    def Constraint_Satisfaction_Check_RQV_Cover(self, folder):
        '''
        Method that checks for any post-allocation margin deficits.
        If there are any deficits, a table is created and saved in the given folder. A net allocation fail amount is also calculated and provided.
        '''

        deficit_list = []

        for j in range(self.margin_set.size):
            if self.margin_set.margins[j].excess_USD < 0:
                deficit_list.append({'counterparty_ID' : self.margin_set.margins[j].counterparty_ID, \
                                     'deficit (USD)': self.margin_set.margins[j].excess_USD})
                
        if len(deficit_list) == 0:
            print('Taker account requirements satisfied.')   
        elif len(deficit_list) > 0:
            deficit_table = pd.DataFrame(deficit_list)
            deficit_table.to_csv(folder + self.entity + ' - Breach Report - RQV Deficits.csv', index=False)
            print('RQV deficits report saved - consider re-run with higher RQV cover tolerance.')
            print('Net allocation fail (USD): ' + '{:,.2f}'.format(deficit_table['deficit (USD)'].sum()))

    def Constraint_Satisfaction_Check_ISIN_Usage(self, folder):
        '''
        Method that checks for any post-allocation security usage breaches.
        If there are any breaches, a table is created and saved in the given folder.
        '''

        breach_list = []

        for i in range(self.inventory.size):
            if self.inventory.securities[i].remaining_notional < 0:
                breach_list.append({'ISIN': self.inventory.securities[i].ISIN, \
                                    'remaining_notional': self.inventory.securities[i].remaining_notional})
                
        if len(breach_list) == 0:
            print('Inventory usage satisfies available notional constraints.')
        elif len(breach_list) > 0:
            breach_table = pd.DataFrame(breach_list)
            breach_table.to_csv(folder + self.entity + ' - Breach Report - Inventory Usage.csv', index=False)
            print('Inventory breach report saved - re-run with higher ISIN usage tolerance.')

    def Constraint_Satisfaction_Check_Concentration_Limits(self, FX_data, folder):
        '''
        Method that checks for any post-allocation concentration limit breaches.
        If there are any breaches, a table is created and saved in the given folder.
        '''

        breach_list = []

        for j in range(self.margin_set.size):
            for k in range(len(self.margin_set.margins[j].concentration_limit_schedule)):

                limit_type = self.margin_set.margins[j].concentration_limit_schedule[k].limit_type
                limit_currency = self.margin_set.margins[j].concentration_limit_schedule[k].limit_currency
                limit_percentage = self.margin_set.margins[j].concentration_limit_schedule[k].limit_percentage
                limit_absolute_value = self.margin_set.margins[j].concentration_limit_schedule[k].limit_absolute_value
                security_currency = self.margin_set.margins[j].concentration_limit_schedule[k].security_currency

                security_currency_index = [i for i, x in enumerate(self.inventory.securities) if x.currency == security_currency]

                if security_currency_index != []:

                    # Compile a subset of result notionals and their associated weights.

                    result_compiler = np.array([])
                    weight_compiler = np.array([])

                    for i in security_currency_index:

                        result_compiler = np.append(result_compiler, self.allocation_result_matrix[i][j])
                        weight_compiler = np.append(weight_compiler, self.allocation_weight_matrix[i][j])

                    breach_amount_USD = 0

                    if limit_type == 'Percentage':

                        if np.dot(result_compiler, weight_compiler) > limit_percentage * self.margin_set.margins[j].RQV_USD:
                            breach_amount_USD = np.dot(result_compiler, weight_compiler) - limit_percentage * self.margin_set.margins[j].RQV_USD

                    elif limit_type == 'Absolute_Value':
                        
                        if np.dot(result_compiler, weight_compiler) > limit_absolute_value * lookup(limit_currency, FX_data, 'currency', 'USD_FX_rate'):
                            breach_amount_USD = np.dot(result_compiler, weight_compiler) - limit_absolute_value * lookup(limit_currency, FX_data, 'currency', 'USD_FX_rate')

                    if breach_amount_USD > 0:
                        breach_list.append({'counterparty_ID': self.margin_set.margins[j].counterparty_ID, \
                                            'limit_type': limit_type, \
                                            'limit_currency': limit_currency, 
                                            'limit_percentage': limit_percentage, \
                                            'limit_absolute_value': limit_absolute_value, \
                                            'security_currency': security_currency, \
                                            'breach_amount_USD': breach_amount_USD})
    
        if len(breach_list) == 0:
            print('Concentration limits satisfied.')
        elif len(breach_list) > 0:
            breach_table = pd.DataFrame(breach_list)
            breach_table.to_csv(folder + self.entity + ' - Breach Report - Concentration Limits.csv', index=False)
            print('Concentration limit breach report saved - consider re-run with higher concentration limit tolerance.')

    # Sufficient Cover Analysis/Exports

    def Export_Allocation_Details(self, folder):
        '''
        Method that creates and exports allocation file.
        '''

        allocation_table = pd.DataFrame(self.allocation_result_matrix, index=[x.ISIN for x in self.inventory.securities], columns=[x.counterparty_ID for x in self.margin_set.margins])
        longbox_table = pd.DataFrame(self.longbox_result_matrix, index=[x.ISIN for x in self.inventory.securities], columns=['Longbox'])
        
        allocation_export_table = pd.concat([allocation_table, longbox_table], axis=1)
        allocation_export_table.to_csv(folder + self.entity + ' - Allocation Details.csv')

        print('\n' + 'Allocation details exported.')

    def Export_Action_Files(self, folder):
        '''
        Method that creates and exports security recall/delivery tables.
        '''

        recall_list = []
        deliver_list = []

        for i in range(self.inventory.size):
            if self.inventory.securities[i].transfer_value < 0:
                recall_list.append({'ISIN': self.inventory.securities[i].ISIN, \
                                    'transfer_value': abs(self.inventory.securities[i].transfer_value)})
            if self.inventory.securities[i].transfer_value > 0:
                deliver_list.append({'ISIN': self.inventory.securities[i].ISIN, \
                                     'transfer_value': abs(self.inventory.securities[i].transfer_value)})
                
        recall_table = pd.DataFrame(recall_list)
        deliver_table = pd.DataFrame(deliver_list)
                
        if len(recall_list) > 0:
            recall_table.to_csv(folder + self.entity + ' - Custodian Recalls.csv', index=False)
            print('Custodian recall file exported.')
            
        if len(deliver_list) > 0:
            deliver_table.to_csv(folder + self.entity + ' - Custodian Deliveries.csv', index=False)
            print('Custodian delivery file exported.')

    # Insufficient Cover Analysis/Exports

    def Insufficient_Cover_Analysis(self, folder):
        '''
        Method that retrieves securities with less than 99% utilisation and exports details.
        Method also prints predicted IA short, saves down deficit report and provides commentary dependent on comparison between unallocated collateral and deficits.
        '''

        # Unallocated collateral export.

        unallocated_list = []

        for i in range(self.inventory.size):
            if self.inventory.securities[i].percentage_utilisation < 0.99:
                unallocated_list.append({'ISIN': self.inventory.securities[i].ISIN, \
                                         'percentage_utilisation': self.inventory.securities[i].percentage_utilisation, \
                                         'remaining_MV_USD': self.inventory.securities[i].remaining_MV_USD})
                
        unallocated_table = pd.DataFrame(unallocated_list)
        total_unallocated_MV_USD = 0
                
        if len(unallocated_list) == 0:
            print('Entire inventory utilised - more collateral required.')
        elif len(unallocated_list) > 0:
            total_unallocated_MV_USD = unallocated_table['remaining_MV_USD'].sum()
            unallocated_table.to_csv(folder + self.entity + ' - Short Allocation - Unallocated Collateral.csv', index=False)
            print('Unallocated collateral report exported.')

        # Predicted IA short and deficit file export.

        deficit_list = []

        for j in range(self.margin_set.size):
            if self.margin_set.margins[j].excess_USD < 0:
                deficit_list.append({'counterparty_ID' : self.margin_set.margins[j].counterparty_ID, \
                                     'deficit (USD)': self.margin_set.margins[j].excess_USD})
                
        deficit_table = pd.DataFrame(deficit_list)
        predicted_IA_short_USD = 0
                
        if len(deficit_list) == 0:
            print('Taker account requirements satisfied.')   
        elif len(deficit_list) > 0:
            predicted_IA_short_USD = deficit_table['deficit (USD)'].sum()
            deficit_table.to_csv(folder + self.entity + ' - Breach Report - RQV Deficits.csv', index=False)
            print('RQV deficits report saved - consider re-run with higher RQV cover tolerance.')
            print('Predicted IA short (USD): ' + '{:,.2f}'.format(predicted_IA_short_USD))

        # Comparison and analysis.

        if total_unallocated_MV_USD > abs(predicted_IA_short_USD):
            print('Value of unallocated collateral is larger than predicted IA allocation short.')
            print('Review unallocated collateral report.')
            print('If unallocated collateral pool is high in value/widely eligible, look to re-run with higher ISIN usage tolerance.')
            print('If unallocated collateral pool is not widely eligible, look to source more collateral.')
        elif total_unallocated_MV_USD > 0:
            print('Review unallocated collateral report. If unallocated collateral pool is not widely eligible, look to source more collateral.')

# General Function List

def lookup(reference, search_table, search_field, target_field):
    '''
    Basic table lookup function.
    '''

    return search_table[search_table[search_field] == reference][target_field].values[0]