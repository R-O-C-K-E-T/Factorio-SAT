from factorio_sat import belt_balancer_net_free_power_of_2
from test.grid_test_case import BaseGridTest


class TestBeltBalancerNetFreePowerOf2(BaseGridTest):
    def test_existing_14_wide_16_to_16(self):
        self.grid = belt_balancer_net_free_power_of_2.create_balancer(14, 16, 8)
        self.grid.block_belts_through_edges((False, True))
        self.grid.prevent_intersection()
        self.grid.enforce_maximum_underground_length()
        self.set_content('''
┌───────────────────────────┐
│→ → → f G l   G → D → f L →│
│→ D l ↓ I G f I L d l ↓ L →│
│→ d l T D h k L f G l k L →│
│→ → → → d D l   ↓ ↑ G f L →│
│→ → → → D d l i ↓ W w ↓ L →│
│→ D l G d f L h ↓ I I T → →│
│→ d l ↑   T → f ↓ L D → → →│
│→ → → h i G l ↓ T → d f L →│
│→ → → f W w K ↓       T → →│
│→ D l ↓ I ↑ s S   i   L D →│
│→ d l T D h k ↓ L h G → d →│
│→ → → → d → f T l   ↑ K L →│
│→ → → → D l ↓     L h T → →│
│→ D l G d f T → f   i L D →│
│→ d l ↑ i ↓ K   ↓ L h G d →│
│→ → → h F g T l T → → h L →│
└───────────────────────────┘
        ''')
        self.assert_sat()

    def test_existing_15_wide_16_to_16(self):
        self.grid = belt_balancer_net_free_power_of_2.create_balancer(15, 16, 8)
        self.grid.block_belts_through_edges((False, True))
        self.grid.prevent_intersection()
        self.grid.enforce_maximum_underground_length()
        self.set_content('''
┌─────────────────────────────┐
│→ → → f H t G → D → f G → → →│
│→ D l ↓ k ↑ I L d l k I   L →│
│→ d l T D h L f G → D → → → →│
│→ → → → d D l k I G d f   L →│
│→ → → → D d l     I K ↓   L →│
│→ D l G d f i L f   k T → → →│
│→ d l ↑ K T h   ↓ L f i G → →│
│→ → → h T D l   ↓   ↓ I ↑ L →│
│→ → → f G d l   T f T D h L →│
│→ D l ↓ I G f   i ↓ L d → → →│
│→ d l T D h k L h T D → → → →│
│→ → → → d D l K G → d f   L →│
│→ → → → D d l T h i K T f L →│
│→ D l G d f     L h T l ↓ L →│
│→ d l ↑ i ↓ K L D → f i T → →│
│→ → → h F g T → d l T h   L →│
└─────────────────────────────┘
        ''')
        self.assert_sat()

    def test_unsat_9_wide_8_to_8(self):
        self.grid = belt_balancer_net_free_power_of_2.create_balancer(9, 8, 8)
        self.grid.block_belts_through_edges((False, True))
        self.grid.prevent_intersection()
        self.grid.enforce_maximum_underground_length()
        self.assert_unsat()

    def test_sat_10_wide_8_to_8(self):
        self.grid = belt_balancer_net_free_power_of_2.create_balancer(10, 8, 4)
        self.grid.block_belts_through_edges((False, True))
        self.grid.prevent_intersection()
        self.grid.enforce_maximum_underground_length()
        self.set_content('''
┌───────────────────┐
│→ → → → → f G → → →│
│→ → l     k ↑ L D →│
│→ → f G l G h L d →│
│→ l ↓ I   ↑ L D → →│
│→ l T D → h L d → →│
│→ → → d f K G → D →│
│→ → l i ↓ T h L d →│
│→ → → h T → → → → →│
└───────────────────┘
        ''')
        self.assert_sat()
