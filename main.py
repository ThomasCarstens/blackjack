import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import random
from collections import defaultdict
import pandas as pd

class BlackjackEnvironment:
    def __init__(self):
        self.reset()
    
    def reset(self):
        # Deal initial cards
        self.player_cards = [self.draw_card(), self.draw_card()]
        self.dealer_cards = [self.draw_card(), self.draw_card()]
        self.done = False
        self.can_double = True
        self.can_surrender = True
        self.bet = 1.0
        self.split_hands = []
        self.current_hand = 0
        return self.get_state()
    
    def draw_card(self):
        """Draw a card (1-13, where 1=Ace, 11=Jack, 12=Queen, 13=King)"""
        return random.randint(1, 13)
    
    def card_value(self, card):
        """Get the value of a card"""
        if card == 1:  # Ace
            return 11
        elif card > 10:  # Face cards
            return 10
        else:
            return card
    
    def hand_value(self, cards):
        """Calculate hand value, handling Aces"""
        value = sum(self.card_value(card) for card in cards)
        aces = sum(1 for card in cards if card == 1)
        
        # Convert Aces from 11 to 1 if needed
        while value > 21 and aces > 0:
            value -= 10
            aces -= 1
        
        return value
    
    def has_usable_ace(self, cards):
        """Check if hand has a usable ace (ace counted as 11)"""
        value = sum(self.card_value(card) for card in cards)
        aces = sum(1 for card in cards if card == 1)
        
        if aces == 0:
            return False
        
        # If we can count at least one ace as 11 without busting
        return value <= 21
    
    def can_split(self):
        """Check if current hand can be split"""
        if len(self.player_cards) != 2:
            return False
        if len(self.split_hands) > 0:  # Already split once
            return False
        return self.card_value(self.player_cards[0]) == self.card_value(self.player_cards[1])
    
    def get_state(self):
        """Get current state (player_sum, dealer_upcard, usable_ace, can_double, can_surrender, can_split)"""
        player_sum = self.hand_value(self.player_cards)
        dealer_upcard = self.card_value(self.dealer_cards[0])
        usable_ace = self.has_usable_ace(self.player_cards)
        return (player_sum, dealer_upcard, usable_ace, self.can_double, self.can_surrender, self.can_split())
    
    def get_valid_actions(self):
        """Get list of valid actions for current state"""
        actions = [0, 1]  # Stand, Hit always available
        
        if self.can_double and len(self.player_cards) == 2:
            actions.append(2)  # Double
        
        if self.can_surrender and len(self.player_cards) == 2:
            actions.append(3)  # Surrender
        
        if self.can_split():
            actions.append(4)  # Split
        
        return actions
    
    def step(self, action):
        """Take action: 0=Stand, 1=Hit, 2=Double, 3=Surrender, 4=Split"""
        if self.done:
            return self.get_state(), 0, True
        
        reward = 0
        
        if action == 0:  # Stand
            reward = self.dealer_play()
            self.done = True
            
        elif action == 1:  # Hit
            self.player_cards.append(self.draw_card())
            player_sum = self.hand_value(self.player_cards)
            self.can_double = False
            self.can_surrender = False
            
            if player_sum > 21:  # Player busts
                reward = -self.bet
                self.done = True
            elif player_sum == 21:  # Player gets 21
                reward = self.dealer_play()
                self.done = True
                
        elif action == 2:  # Double
            if self.can_double and len(self.player_cards) == 2:
                self.bet *= 2
                self.player_cards.append(self.draw_card())
                player_sum = self.hand_value(self.player_cards)
                
                if player_sum > 21:  # Player busts
                    reward = -self.bet
                else:
                    reward = self.dealer_play()
                self.done = True
            else:
                # Invalid action, treat as hit
                return self.step(1)
                
        elif action == 3:  # Surrender
            if self.can_surrender and len(self.player_cards) == 2:
                reward = -0.5 * self.bet
                self.done = True
            else:
                # Invalid action, treat as stand
                return self.step(0)
                
        elif action == 4:  # Split
            if self.can_split():
                # Split the hand
                card1 = self.player_cards[0]
                card2 = self.player_cards[1]
                
                # First hand
                self.player_cards = [card1, self.draw_card()]
                # Second hand stored for later
                self.split_hands = [[card2, self.draw_card()]]
                
                self.can_double = True  # Can double after split
                self.can_surrender = False  # Cannot surrender after split
                
                # Continue with first hand
                player_sum = self.hand_value(self.player_cards)
                if player_sum == 21:
                    # Auto-stand on 21
                    reward = self.play_split_hands()
                    self.done = True
            else:
                # Invalid action, treat as hit
                return self.step(1)
        
        return self.get_state(), reward, self.done
    
    def play_split_hands(self):
        """Play out split hands with basic strategy"""
        total_reward = 0
        
        # Play first hand (current hand)
        if self.hand_value(self.player_cards) <= 21:
            total_reward += self.dealer_play_single_hand(self.player_cards)
        else:
            total_reward -= self.bet
        
        # Play split hands
        for split_hand in self.split_hands:
            # Simple strategy for split hands
            while self.hand_value(split_hand) < 17:
                split_hand.append(self.draw_card())
            
            if self.hand_value(split_hand) <= 21:
                total_reward += self.dealer_play_single_hand(split_hand)
            else:
                total_reward -= self.bet
        
        return total_reward
    
    def dealer_play_single_hand(self, player_hand):
        """Dealer plays against a single hand"""
        dealer_cards = self.dealer_cards.copy()
        
        while self.hand_value(dealer_cards) < 17:
            dealer_cards.append(self.draw_card())
        
        player_sum = self.hand_value(player_hand)
        dealer_sum = self.hand_value(dealer_cards)
        
        if dealer_sum > 21:  # Dealer busts
            return self.bet
        elif player_sum > dealer_sum:
            return self.bet
        elif player_sum < dealer_sum:
            return -self.bet
        else:
            return 0  # Tie
    
    def dealer_play(self):
        """Dealer plays according to rules (hit on soft 17)"""
        if len(self.split_hands) > 0:
            return self.play_split_hands()
        
        while self.hand_value(self.dealer_cards) < 17:
            self.dealer_cards.append(self.draw_card())
        
        player_sum = self.hand_value(self.player_cards)
        dealer_sum = self.hand_value(self.dealer_cards)
        
        if dealer_sum > 21:  # Dealer busts
            return self.bet
        elif player_sum > dealer_sum:
            return self.bet
        elif player_sum < dealer_sum:
            return -self.bet
        else:
            return 0  # Tie

class BlackjackQLearning:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.3):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(5))  # 5 actions: Stand, Hit, Double, Surrender, Split
        self.env = BlackjackEnvironment()
        self.action_names = ['Stand', 'Hit', 'Double', 'Surrender', 'Split']
        self.action_letters = ['S', 'H', 'D', 'R', 'P']
    
    def get_action(self, state, training=True):
        """Get action using epsilon-greedy policy with action masking"""
        valid_actions = self.env.get_valid_actions()
        
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            # Only consider valid actions
            q_values = self.q_table[state]
            valid_q_values = [(action, q_values[action]) for action in valid_actions]
            return max(valid_q_values, key=lambda x: x[1])[0]
    
    def train(self, episodes=5000000):  # 10x more experiments
        """Train the Q-learning agent"""
        wins = 0
        losses = 0
        ties = 0
        total_reward = 0
        
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = self.get_action(state, training=True)
                next_state, reward, done = self.env.step(action)
                episode_reward += reward
                
                # Q-learning update with action masking
                valid_next_actions = self.env.get_valid_actions() if not done else [0, 1]
                if valid_next_actions:
                    best_next_q = max(self.q_table[next_state][a] for a in valid_next_actions)
                else:
                    best_next_q = 0
                
                td_target = reward + self.gamma * best_next_q
                td_error = td_target - self.q_table[state][action]
                self.q_table[state][action] += self.lr * td_error
                
                state = next_state
            
            total_reward += episode_reward
            
            # Track results
            if episode_reward > 0:
                wins += 1
            elif episode_reward < 0:
                losses += 1
            else:
                ties += 1
            
            # Decay epsilon and report progress
            if episode % 100000 == 0 and episode > 0:
                self.epsilon = max(0.01, self.epsilon * 0.98)
                avg_reward = total_reward / episode
                print(f"Episode {episode}: Win rate = {wins/(wins+losses+ties):.3f}, "
                      f"Avg reward = {avg_reward:.3f}, Epsilon = {self.epsilon:.3f}")
        
        final_win_rate = wins / (wins + losses + ties)
        avg_reward = total_reward / episodes
        print(f"Training completed. Final win rate: {final_win_rate:.3f}, Avg reward: {avg_reward:.3f}")
    
    def get_policy(self):
        """Extract policy from Q-table"""
        policy = {}
        for state in self.q_table:
            # Simulate environment state for action masking
            temp_env = BlackjackEnvironment()
            temp_env.player_cards = [1, 1]  # Dummy cards
            temp_env.dealer_cards = [1, 1]
            temp_env.can_double = state[3] if len(state) > 3 else True
            temp_env.can_surrender = state[4] if len(state) > 4 else True
            
            # Get valid actions for this state
            valid_actions = [0, 1]  # Stand and Hit always valid
            if len(state) > 3 and state[3]:  # can_double
                valid_actions.append(2)
            if len(state) > 4 and state[4]:  # can_surrender
                valid_actions.append(3)
            if len(state) > 5 and state[5]:  # can_split
                valid_actions.append(4)
            
            # Find best valid action
            q_values = self.q_table[state]
            valid_q_values = [(action, q_values[action]) for action in valid_actions]
            policy[state] = max(valid_q_values, key=lambda x: x[1])[0]
        
        return policy

def create_strategy_chart(agent):
    """Create comprehensive strategy chart with all actions"""
    policy = agent.get_policy()
    
    # Create matrices for different scenarios
    strategy_no_ace = np.full((10, 10), -1, dtype=int)  # Player sums 12-21, Dealer 2-11
    strategy_ace = np.full((10, 10), -1, dtype=int)
    strategy_pairs = np.full((10, 10), -1, dtype=int)  # For pairs (can split)
    
    # Fill in the strategy matrices
    for player_sum in range(12, 22):
        for dealer_card in range(2, 12):
            dealer_idx = 9 if dealer_card == 11 else dealer_card - 2
            player_idx = player_sum - 12
            
            # Regular hands (no usable ace, no split)
            state = (player_sum, dealer_card, False, True, True, False)
            if state in policy:
                strategy_no_ace[player_idx, dealer_idx] = policy[state]
            
            # Soft hands (with usable ace)
            state = (player_sum, dealer_card, True, True, True, False)
            if state in policy:
                strategy_ace[player_idx, dealer_idx] = policy[state]
            
            # Pairs (can split)
            if player_sum % 2 == 0 and player_sum <= 20:  # Even sums that could be pairs
                state = (player_sum, dealer_card, False, True, True, True)
                if state in policy:
                    strategy_pairs[player_idx, dealer_idx] = policy[state]
    
    # Create action labels and colors
    action_labels = ['S', 'H', 'D', 'R', 'P']
    colors = ['red', 'lightgreen', 'yellow', 'orange', 'purple']
    
    # Create custom colormap
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors)
    
    # Create the plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot hard hands (no usable ace)
    im1 = ax1.imshow(strategy_no_ace, cmap=cmap, vmin=0, vmax=4, aspect='auto')
    ax1.set_title('Hard Hands (No Usable Ace)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Dealer Upcard')
    ax1.set_ylabel('Player Sum')
    ax1.set_xticks(range(10))
    ax1.set_xticklabels(['2','3','4','5','6','7','8','9','10','A'])
    ax1.set_yticks(range(10))
    ax1.set_yticklabels(range(12, 22))
    
    # Add action letters to cells
    for i in range(10):
        for j in range(10):
            if strategy_no_ace[i, j] >= 0:
                ax1.text(j, i, action_labels[strategy_no_ace[i, j]], 
                        ha='center', va='center', fontweight='bold')
    
    # Plot soft hands (with usable ace)
    im2 = ax2.imshow(strategy_ace, cmap=cmap, vmin=0, vmax=4, aspect='auto')
    ax2.set_title('Soft Hands (With Usable Ace)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Dealer Upcard')
    ax2.set_ylabel('Player Sum')
    ax2.set_xticks(range(10))
    ax2.set_xticklabels(['2','3','4','5','6','7','8','9','10','A'])
    ax2.set_yticks(range(10))
    ax2.set_yticklabels(range(12, 22))
    
    # Add action letters to cells
    for i in range(10):
        for j in range(10):
            if strategy_ace[i, j] >= 0:
                ax2.text(j, i, action_labels[strategy_ace[i, j]], 
                        ha='center', va='center', fontweight='bold')
    
    # Plot pairs
    im3 = ax3.imshow(strategy_pairs, cmap=cmap, vmin=0, vmax=4, aspect='auto')
    ax3.set_title('Pairs (Can Split)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Dealer Upcard')
    ax3.set_ylabel('Player Pair')
    ax3.set_xticks(range(10))
    ax3.set_xticklabels(['2','3','4','5','6','7','8','9','10','A'])
    ax3.set_yticks(range(10))
    ax3.set_yticklabels(['6s','7s','8s','9s','10s','Js','Qs','Ks','As',''])
    
    # Add action letters to cells
    for i in range(10):
        for j in range(10):
            if strategy_pairs[i, j] >= 0:
                ax3.text(j, i, action_labels[strategy_pairs[i, j]], 
                        ha='center', va='center', fontweight='bold')
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i], label=f'{action_labels[i]} - {agent.action_names[i]}') 
                      for i in range(5)]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=5)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()

def create_action_distribution_chart(agent):
    """Create chart showing action distribution across different scenarios"""
    policy = agent.get_policy()
    
    action_counts = defaultdict(int)
    total_states = 0
    
    for state, action in policy.items():
        action_counts[action] += 1
        total_states += 1
    
    # Create pie chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Action distribution pie chart
    actions = list(action_counts.keys())
    counts = list(action_counts.values())
    labels = [f"{agent.action_names[a]} ({agent.action_letters[a]})" for a in actions]
    
    ax1.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Overall Action Distribution')
    
    # Action frequency by dealer upcard
    dealer_action_matrix = np.zeros((10, 5))  # 10 dealer cards, 5 actions
    dealer_totals = np.zeros(10)
    
    for state, action in policy.items():
        if len(state) >= 2:
            dealer_card = state[1]
            dealer_idx = 9 if dealer_card == 11 else dealer_card - 2
            if 0 <= dealer_idx < 10:
                dealer_action_matrix[dealer_idx, action] += 1
                dealer_totals[dealer_idx] += 1
    
    # Normalize by dealer card frequency
    for i in range(10):
        if dealer_totals[i] > 0:
            dealer_action_matrix[i] /= dealer_totals[i]
    
    im = ax2.imshow(dealer_action_matrix.T, cmap='Blues', aspect='auto')
    ax2.set_title('Action Frequency by Dealer Upcard')
    ax2.set_xlabel('Dealer Upcard')
    ax2.set_ylabel('Action')
    ax2.set_xticks(range(10))
    ax2.set_xticklabels(['2','3','4','5','6','7','8','9','10','A'])
    ax2.set_yticks(range(5))
    ax2.set_yticklabels([f"{agent.action_letters[i]}" for i in range(5)])
    
    # Add colorbar
    plt.colorbar(im, ax=ax2, label='Frequency')
    
    plt.tight_layout()
    plt.show()

def evaluate_agent(agent, episodes=100000):
    """Evaluate the trained agent"""
    wins = 0
    losses = 0
    ties = 0
    total_reward = 0
    action_counts = defaultdict(int)
    
    for _ in range(episodes):
        state = agent.env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.get_action(state, training=False)
            action_counts[action] += 1
            state, reward, done = agent.env.step(action)
            episode_reward += reward
        
        total_reward += episode_reward
        
        if episode_reward > 0:
            wins += 1
        elif episode_reward < 0:
            losses += 1
        else:
            ties += 1
    
    win_rate = wins / episodes
    avg_reward = total_reward / episodes
    
    print(f"Evaluation over {episodes} games:")
    print(f"Wins: {wins} ({wins/episodes:.3f})")
    print(f"Losses: {losses} ({losses/episodes:.3f})")
    print(f"Ties: {ties} ({ties/episodes:.3f})")
    print(f"Win rate: {win_rate:.3f}")
    print(f"Average reward per game: {avg_reward:.3f}")
    
    print(f"\nAction usage during evaluation:")
    total_actions = sum(action_counts.values())
    for action, count in sorted(action_counts.items()):
        print(f"  {agent.action_names[action]} ({agent.action_letters[action]}): {count} ({count/total_actions:.3f})")
    
    return win_rate, avg_reward

if __name__ == "__main__":
    print("Training Enhanced Blackjack Q-Learning Agent with 5 Actions...")
    print("Actions: Stand (S), Hit (H), Double (D), Surrender (R), Split (P)")
    
    # Create and train agent
    agent = BlackjackQLearning(learning_rate=0.1, discount_factor=0.95, epsilon=0.4)
    agent.train(episodes=100000000)  # 10x more experiments
    
    # Evaluate agent
    print("\nEvaluating trained agent...")
    win_rate, avg_reward = evaluate_agent(agent, episodes=100000)
    
    # Create visualizations
    print("\nCreating comprehensive strategy charts...")
    create_strategy_chart(agent)
    
    print("Creating action distribution analysis...")
    create_action_distribution_chart(agent)
    
    print("Done! The enhanced agent has been trained with 10x more experiments and comprehensive visualizations are displayed.")