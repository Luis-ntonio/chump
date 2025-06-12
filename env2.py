import numpy as np
import gym
from gym import spaces
from math import dist
import math
from itertools import product
import random

class InventoryEnv(gym.Env):
    def __init__(self, num_customers=5, grid_size=10,capacity_supplier=30,capacity_customers=20,ss_customers=0,ss_supplier=0,num_trucks=2,scale_supplier=10.0,scale_customers=8.0,benchmark_location=1,max_truck_capacity=20,custom_reward=None,holding_cost_supplier=0.5,holding_cost_customer=1.0,backorder_cost=10.0):
        """Inventory Management Environment for Reinforcement Learning."""
        super(InventoryEnv, self).__init__()

        self.num_customers = num_customers
        self.grid_size = grid_size

        # Fixed parameters
        self.capacity_supplier = capacity_supplier
        self.capacity_customers = [capacity_customers for _ in range(num_customers)]

        #Right now is 0 but it can be modified to simulate to restock when reaching ss
        self.SS_customers = [ss_customers for _ in range(num_customers)]
        self.SS_supplier = ss_supplier
        self.max_delivery = max_truck_capacity

        self.M = num_trucks # number of trucks
        self.m = 15 # max time per truck
        self.delta = 1.0
        self.service_time = 0.5
        self.market_price = 1.0

        # Configurable gamma demmand parameters
        self.shape_supplier = 1.0
        self.scale_supplier = scale_supplier
        self.shape_customer = 1.0
        self.scale_customer = scale_customers

        # Cost parameters holding
        self.holding_cost_supplier = holding_cost_supplier
        self.holding_cost_customer = holding_cost_customer

        #Cost parameters
        self.backorder_cost = backorder_cost
        self.transport_cost = 1.0
        self.fixed_truck_cost = 1.0

        # Locations of supplier
        self.supplier_location = (5, 5)
        self.custom_reward = custom_reward

        if benchmark_location==1:
        #Benchmark Locations
            self.customer_locations = [(2, 3), 
            (7, 1), 
            (4, 8)
              
            ]
        else:
        #Locations of customers(random assigned)
            self.customer_locations = [
            (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
            for _ in range(num_customers)
            ]

        # Observation space: supplier + customer inventories
        low = np.array([0] * (1 + num_customers), dtype=np.int32) #0 for every action 
        high = np.array([self.capacity_supplier] + self.capacity_customers, dtype=np.int32) # max capacity of supplier and max capacity of supplier
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        

        # Acción: [market_sale, delivery_client_1, ..., delivery_client_n]

        self.action_space = spaces.MultiDiscrete(
        [self.capacity_supplier+1] + [capacity_customers+1 for _ in range(num_customers)]
        )


        self.reset()

    def reset(self):
        self.inventory_supplier = 12
        self.inventory_customers = [5  for _ in range(self.num_customers)]
        self.state = [self.inventory_supplier] + self.inventory_customers
        return np.array(self.state, dtype=np.int32)
    
    def rutas_combinadas(self,clientes, proveedor, C, m, servicio, delta=1.0):
        """
        Returns all possible combinations of truck assignments to customers
        (allowing repetitions), using indices starting from 1.

        Args:
            clientes: list of tuple with coordinates of the clients
            proveedor: tuple with coordinates of the provider
            C: (int) number of trucks
            m: (float) maximum time per trip
            servicio: (float) service time per client
            delta: (float) time per unit of distance

        Returns:
            list of tuples, where each tuple contains indices of clients
        """
        clientes_viables = []
        for idx, cliente in enumerate(clientes):
            distancia = np.linalg.norm(np.array(proveedor) - np.array(cliente))
            tiempo = 2 * distancia * delta + servicio
            if tiempo <= m:
                clientes_viables.append(idx + 1)  # <-- index starts from 1

        combinaciones = list(product(clientes_viables, repeat=C))
        return combinaciones

    def posible_actions_stock(self,state, capacity):
        """
        Generate possible actions for stock delivery.

        Args:
            state (np.array): Current state (first position is market, rest are clients)
            capacity (int): Maximum capacity of Store

        Returns:
            list of np.array: Possible actions
        """
        I_s = state[0]
        stock_clientes = state[1:]
        num_customers = len(stock_clientes)

        actions = []

        for combo in product(range(I_s + 1), repeat=num_customers + 1):
            total_entregado = sum(combo)
            entrega_clientes = combo[1:]

            if total_entregado <= I_s and all(
                entrega_clientes[i] + stock_clientes[i] <= capacity
                for i in range(num_customers)
            ):
                actions.append(np.array(combo, dtype=np.int32))
        
        return actions

    def filtrar_acciones_con_capacidad(self,actions, clientes_alcanzables, Q2, C):
        """
        Filtra acciones que pueden realizarse considerando:
        - Clientes alcanzables (índices desde 1)
        - Capacidad de cada camión Q2
        - Número total de camiones disponibles C
        """
        acciones_filtradas = []

        for accion in actions:
            entregas = accion[1:]  # sin el mercado
            clientes_con_entrega = [i + 1 for i, q in enumerate(entregas) if q > 0]

            # Filtro 1: ¿Todos los clientes son alcanzables?
            if not all(c in clientes_alcanzables for c in clientes_con_entrega):
                continue

            # Filtro 2: ¿Cuántos camiones necesitaríamos en total?
            camiones_necesarios = sum(math.ceil(q / Q2) for q in entregas)

            if camiones_necesarios <= C:
                acciones_filtradas.append(accion)

        return acciones_filtradas
   
   
    def action_state_space(self):
        rutas = self.rutas_combinadas(self.customer_locations, self.supplier_location, self.M, self.m, self.service_time, self.delta)
        acciones = self.posible_actions_stock(self.state, self.capacity_customers[0])
        clientes_alcanzables = list(set(cliente for ruta in rutas for cliente in ruta))
        acciones_filtradas = self.filtrar_acciones_con_capacidad(acciones, clientes_alcanzables, self.max_delivery, self.M)
        return acciones_filtradas


    def total_transportation_cost(self,accion, clientes, proveedor, Q2, F, delta=1.0):
        """
    
        Calculate the total cost of an action considering:
        - Fixed cost for each truck used (F)
        - Variable cost based on distance for each trip

        Args:
            accion(array: vector [market_sale, c1, c2, ..., cn])
            clients(list of tuple): coordinates of the clients
            supplier(tuple): coordinates of the supplier
            Q2(int): capacity per truck
            F(float): fixed cost per truck used
            delta(float): cost per unit of distance (default 1)

        Returns:
            float: total cost of the action
        """
        total_cost = 0.0
        camiones_usados = 0

        entregas = accion[1:]  # omitimos el mercado
        for i, cantidad in enumerate(entregas):
            if cantidad == 0:
                continue
            
            coord_cliente = clientes[i]
            distancia = np.linalg.norm(np.array(coord_cliente) - np.array(proveedor))
            viajes = math.ceil(cantidad / Q2)

            # Por cada viaje: ida y vuelta
            costo_viajes = viajes * (2 * distancia * delta)
            costo_fijo = viajes * F

            total_cost += costo_fijo + costo_viajes
            camiones_usados += viajes

        return float(total_cost)
    
   

    def normalized_reward(self, mkt_revenue,cost):
        # Normalizar profit respecto al baseline
        profit = mkt_revenue - cost
        if self.custom_reward is not None and self.custom_reward != 0:
            normalized_profit = (profit + self.custom_reward) / self.custom_reward
        else:
            normalized_profit = profit / 1  # Escalar arbitrariamente

        return normalized_profit
        

    def get_action(self):

        # there are some missing elements
        # 0) we do not check yet if the inventory at the supplier is sufficient to send to the customers
        # 1) there is no action yet of selling to the outside market
        # 2) there is check yet if the truck capacity is sufficient
        # 3) there is possibility yet that the truck can locations muliple times
        # 4) associated with 3 - there is no check yet that the shift duration of a truck is respected if it does multiple trips.
    
        action = np.zeros(self.nCust)
        tmp_array = self.orderUpTo - self.inventories[1:]

        for i in range(0, self.nCust):
            if  tmp_array[i] > 0:
               action[i] = 1

        return action
    

    def full_truck_actions(self, acciones_filtradas, Q2, C):
        """
        Filtra acciones donde cada cliente que recibe entrega debe recibir exactamente Q2,
        es decir, cada camión va lleno y sirve a un solo cliente.

        Args:
            acciones_filtradas (list of np.array): Acciones válidas de action_state_space()
            Q2 (int): Capacidad del camión
            C (int): Número máximo de camiones

        Returns:
            list of np.array: Acciones válidas bajo la condición de "camión lleno por tienda"
        """
        acciones_validas = []

        for accion in acciones_filtradas:
            entregas = accion[1:]  # sin el mercado
            clientes_atendidos = [q for q in entregas if q > 0]

            # Cada cliente debe recibir exactamente Q2 unidades
            if all(q == Q2 for q in clientes_atendidos):
                # Número total de camiones usados no debe superar C
                if len(clientes_atendidos) <= C:
                    acciones_validas.append(accion)

        return acciones_validas

    
    
    def step(self, action):
        # Simulate random supplier production
        
        #Equal the following variables to the action specified
        market_sale = action[0]         # how much to sell to the market
        deliveries = list(action[1:])   # how much to deliver to customers
        

        # Validate if the action can be performed according to the current state of trucks and capacity
        deliveries = [min(e, cap - inv) for e, cap, inv in zip(deliveries, self.capacity_customers, self.state[1:])]
        total_deliveries = sum(deliveries)

        # Simulate random demand
        demands = [int(np.random.gamma(self.shape_customer, self.scale_customer)) for _ in range(self.num_customers)]

        cost_transport_total = self.total_transportation_cost(action, self.customer_locations, self.supplier_location, self.max_delivery, self.fixed_truck_cost, self.delta)

        #Decrease inventory of supplier
        self.state[0] -= market_sale
        market_revenue = market_sale * self.market_price #Market revenue from sales
        
        #Decrease inventory of supplier with deliveries
        self.state[0] -= total_deliveries


        #Update inventory of customer with deliveries
        for i in range(self.num_customers):
            self.state[1 + i] = min(self.capacity_customers[i], self.state[1 + i] + deliveries[i])

        #Print Post-Action State
        post_action_state = self.state.copy()
        # Calculate costs supplier
        holding_supplier = self.state[0] * self.holding_cost_supplier

        #Calculate Backorders Quantity
        backorders_quantity = ([
            max(0, demand - (inv)) 
            for demand, inv in zip(demands, self.state[1:])
        ])
        #Calulcate backorders cost
        backorders_cost = sum([
            max(0, demand - (inv)) * self.backorder_cost
            for demand, inv in zip(demands, self.state[1:])
        ])

        # Update inventory of customers with demand
        for i in range(self.num_customers):
            self.state[1 + i] = max(0, self.state[1 + i] - demands[i])

        #Calculate holding cost of customers 
        holding_customers = sum([inv * self.holding_cost_customer for inv in self.state[1:]])
        
        #Calculate total cost 
        total_cost = holding_supplier + holding_customers + backorders_cost + cost_transport_total
        reward = self.normalized_reward(market_revenue, total_cost)
        



        production = int(np.random.gamma(self.shape_supplier, self.scale_supplier))
        #Increase inventory of supplier with production
        self.state[0] = max(0,min(self.capacity_supplier, self.state[0] + production))
        pre_state = self.state.copy()  # Store the pre-action state for info
        

        # Calcular service rate: entregas efectivas / demanda total (evitando división entre 0)
        total_demand = sum(demands)
        total_delivered = sum(deliveries)
        if sum(backorders_quantity) > 0:
            service_level =  0
        else:
            service_level = 1
        
        # Service level per store: 1 if no backorder, 0 if backorder occurred
        service_level_per_store = [0 if b > 0 else 1 for b in backorders_quantity]
        
        done = False  # can change if you define a horizon
        info = {
            'demand': demands,
            'backorders': backorders_quantity,
            'deliveries_efective': deliveries,
            'market_sold': market_sale,
            'costs': total_cost,
            'market_revenue': market_revenue,
            'production': production,
            'service_level': service_level,
            'pre_state': pre_state,
            'post_action_state': post_action_state,
            'service_level_per_store': service_level_per_store
        }


        return np.array(self.state, dtype=np.int32), reward, done, info
    
'''
envi = InventoryEnv(num_customers=2, grid_size=10, capacity_supplier=30, capacity_customers=20, ss_customers=0, ss_supplier=0, num_trucks=2, scale_supplier=10.0, scale_customers=8.0, benchmark_location=1)
obs = envi.reset()
print("Initial State", obs)

for step in range(5):
    action = random.choice(envi.action_state_space())
    obs, reward, done, info = envi.step(action)
    print(f"\ Step {step + 1}")
    print("ACTION:", action)
    print('POST-ACTION STATE:',info['post_action_state'] )
    print("DEMAND:", info["demand"])
    print("BACKORDERS QUANTITY:", info["backorders"])
    print("NEW STATE:", obs)
    print("REWARD:", reward)
|'''