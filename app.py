import json
import os
import gradio as gr
import pandas as pd
import re
from openai import OpenAI
import requests
import sys
from typing import List, Dict, Any, Tuple
import base64

# Add this function at the top of your file
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Get the base64 string for your logo
logo_base64 = get_image_base64("Logo.png")

# Load the JSON data
with open('premium_collections.json', 'r') as f:
    premium_collections = json.load(f)
    
with open('clothing.json', 'r') as f:
    clothing = json.load(f)

#with open('products.json', 'r') as f:
#    products = json.load(f)

# Combine both datasets and tag them with their source
for item in premium_collections:
    item['source'] = 'premium_collections'
for item in clothing:
    item['source'] = 'clothing'

#for item in products:
#    item['source'] = 'products'

#all_items = products
all_items = premium_collections + clothing
# Function to normalize price strings to float
def normalize_price(price_str):
    if not price_str:
        return None
    
    # Handle ranges like "$8.50 – $28.00"
    if '–' in price_str or '-' in price_str:
        parts = re.split(r'–|-', price_str)
        # Take the lower price for calculation
        price_str = parts[0].strip()
    
    # Extract the numeric value
    match = re.search(r'(\d+\.\d+|\d+)', price_str)
    if match:
        return float(match.group(1))
    return None

# Process items to have normalized prices
for item in all_items:
    if item.get('price'):
        item['normalized_price'] = normalize_price(item['price'])

# Define all the dropdown options
AGE_GROUPS = ["Choose an option", "<18", "18-30", "30-40", "40-50", "50-60", ">60"]
GIFT_OCCASIONS = [
    "Choose an option",
    "Festive Celebration", 
    "Long Service Award", 
    "Corporate Milestones", 
    "Onboarding", 
    "Christmas/Year-End Celebration", 
    "Annual Dinner & Dance", 
    "All The Best!", 
    "Others"
]
COLOR_THEMES = [
    "Choose an option",
    "Black", "White", "Off-White", "Brown", "Red", "Blue", "Gray", 
    "Gold", "Yellow", "Purple", "Pink", "Green", "Silver", 
    "Orange", "Multi-color", "Transparent"
]
JOB_FUNCTIONS = [
    "Choose an option",
    "C-Suite", 
    "Sales & Business Development", 
    "Finance", 
    "Operations", 
    "Human Resource", 
    "Engineering", 
    "Information Technology", 
    "Marketing & Communications", 
    "Others"
]
GENDERS = ["Choose an option", "Male", "Female", "does not really matter"]

# Budget options for the new interface
BUDGET_RANGES = [
    "Below S$10",
    "S$10 to S$20",
    "S$20 to S$35",
    "S$35 to S$55",
    "S$55 to S$80"
]

# Configure API keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Default Ollama URL

class BudgetAgent:
    def __init__(self, items, model="deepseek-r1:32b"):
        self.items = items
        self.model = model
    
    def calculate_bundle(self, min_budget: float, max_budget: float, selected_items: list) -> tuple:
        """
        Calculate if the selected items fit within the budget range.
        Returns: (fits_budget, total_cost, explanation)
        """
        # Filter out items without valid prices
        valid_items = [item for item in selected_items if item.get('normalized_price') is not None]
        
        if not valid_items:
            return False, 0, "No items with valid prices were selected."
        
        total_cost = sum(item['normalized_price'] for item in valid_items)
        
        # Check if total fits within budget range
        fits_budget = min_budget <= total_cost <= max_budget
        
        # Create explanation
        item_details = [f"{item['name']} (S${item['normalized_price']:.2f})" for item in valid_items]
        explanation = f"Total cost: S${total_cost:.2f} for items: {', '.join(item_details)}. "
        
        if fits_budget:
            explanation += f"This bundle is within your budget range of S${min_budget:.2f} to S${max_budget:.2f}."
        else:
            if total_cost < min_budget:
                explanation += f"This bundle is below your minimum budget of S${min_budget:.2f} by S${min_budget - total_cost:.2f}."
            else:
                explanation += f"This bundle exceeds your maximum budget of S${max_budget:.2f} by S${total_cost - max_budget:.2f}."
        
        return fits_budget, total_cost, explanation
    
    def query_ollama(self, prompt):
        """Query the local Ollama model"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            response = requests.post(OLLAMA_API_URL, json=payload)
            if response.status_code == 200:
                return response.json().get("response", "Error: No response from Ollama")
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Error connecting to Ollama: {str(e)}"

    # Update the BudgetAgent's optimize_bundle method to respect the desired number of items
    def optimize_bundle(self, min_budget: float, max_budget: float, items: list, criteria: str = None, num_items: str = "Any number") -> list:
        """
        Optimize item selection to fit within budget range while maximizing value.
        The criteria parameter is optional for additional filtering logic.
        The num_items parameter allows specifying the desired number of items in the bundle.
        """
        # Filter out items without prices
        valid_items = [item for item in items if item.get('normalized_price') is not None]
        
        if not valid_items:
            return []
        
        # Sort items by price (highest first) for initial selection
        valid_items.sort(key=lambda x: x.get('normalized_price', 0), reverse=True)
        
        # Calculate current total
        current_total = sum(item.get('normalized_price', 0) for item in valid_items)
        
        # If over budget, remove expensive items
        if current_total > max_budget:
            while valid_items and current_total > max_budget:
                removed_item = valid_items.pop(0)  # Remove the most expensive item
                current_total -= removed_item.get('normalized_price', 0)
        
        # If under minimum budget, try to add more items
        if current_total < min_budget:
            remaining_items = [item for item in self.items if
                              item.get('normalized_price') is not None and
                              item not in valid_items]
            
            # Sort by price (ascending) to add cheaper items first
            remaining_items.sort(key=lambda x: x.get('normalized_price', 0))
            
            for item in remaining_items:
                item_price = item.get('normalized_price', 0)
                if current_total + item_price <= max_budget:
                    valid_items.append(item)
                    current_total += item_price
                    if current_total >= min_budget:
                        break
        
        # Now adjust based on the desired number of items
        if num_items != "Any number":
            desired_count = 0
            
            if num_items == "1 item only":
                desired_count = 1
            elif num_items == "2 items":
                desired_count = 2
            elif num_items == "3 items":
                desired_count = 3
            elif num_items == "4 items":
                desired_count = 4
            elif num_items == "5 or more items":
                desired_count = 5  # We'll use 5 as the minimum for this category
            
            current_count = len(valid_items)
            
            if num_items == "5 or more items" and current_count < desired_count:
                # Add more items to reach the minimum of 5 for this category
                remaining_items = [item for item in self.items if
                                 item.get('normalized_price') is not None and
                                 item not in valid_items]
                
                # Sort by price (ascending) to add cheaper items first
                remaining_items.sort(key=lambda x: x.get('normalized_price', 0))
                
                for item in remaining_items:
                    item_price = item.get('normalized_price', 0)
                    if current_total + item_price <= max_budget:
                        valid_items.append(item)
                        current_total += item_price
                        current_count += 1
                        
                        # Stop when we reach the desired count
                        if current_count >= desired_count:
                            break
            
            elif current_count > desired_count:
                # Too many items, remove the lowest value ones
                valid_items.sort(key=lambda x: x.get('normalized_price', 0))
                
                while len(valid_items) > desired_count:
                    removed_item = valid_items.pop(0)  # Remove the cheapest item
                    current_total -= removed_item.get('normalized_price', 0)
                    
            elif current_count < desired_count:
                # Too few items, add more while staying under budget
                remaining_items = [item for item in self.items if
                                 item.get('normalized_price') is not None and
                                 item not in valid_items]
                
                # Sort by price (ascending) to add cheaper items first
                remaining_items.sort(key=lambda x: x.get('normalized_price', 0))
                
                for item in remaining_items:
                    item_price = item.get('normalized_price', 0)
                    if current_total + item_price <= max_budget:
                        valid_items.append(item)
                        current_total += item_price
                        current_count += 1
                        
                        # Stop when we reach the desired count
                        if current_count >= desired_count:
                            break
        
        # Return the optimized selection
        return valid_items

    
    def adjust_bundle_to_fit_total_budget(self, bundle_items: list, min_budget: float, max_budget: float, total_budget: float) -> tuple:
        """
        Adjust the number of bundle packages to fit within the total budget.
        Returns: (num_packages, total_cost, explanation)
        """
        if not bundle_items:
            return 0, 0, "No items in the bundle to calculate."
        
        # Calculate the cost of a single bundle
        bundle_cost = sum(item.get('normalized_price', 0) for item in bundle_items)
        
        if bundle_cost == 0:
            return 0, 0, "Bundle has no valid price information."
            
        # Calculate how many complete bundles fit within the total budget
        max_packages = int(total_budget / bundle_cost)
        
        if max_packages == 0:
            return 0, 0, f"The bundle cost (S${bundle_cost:.2f}) exceeds your total budget of S${total_budget:.2f}."
        
        total_cost = bundle_cost * max_packages
        
        # Check if the bundle meets minimum budget requirements
        bundle_meets_min = bundle_cost >= min_budget
        
        explanation = f"Each bundle costs S${bundle_cost:.2f}. "
        
        if not bundle_meets_min:
            explanation += f"Note: The bundle is below your minimum item budget of S${min_budget:.2f}. "
        
        explanation += f"You can purchase {max_packages} complete bundle(s) for a total of S${total_cost:.2f}, "
        explanation += f"which is within your total budget of S${total_budget:.2f}."
        
        if total_budget - total_cost > 0:
            explanation += f" You will have S${total_budget - total_cost:.2f} remaining."
        
        return max_packages, total_cost, explanation

# In the SelectionAgent class, modify the select_items method to include total_budget parameter

# In the SelectionAgent class, modify the select_items method to include total_budget parameter

class SelectionAgent:
    def __init__(self, items):
        self.items = items
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        # Define weights for different criteria
        self.weights = {
            "query": 0.30,              # Free text query (30%)
            "budget": {                 # Budget considerations (30%)
                "price_range": 0.15,    # Budget range per package
                "total_budget": 0.10,   # Total budget
                "item_count": 0.13      # Number of items per package
            },
            "demographics": {           # Demographics (20%)
                "age_group": 0.04,      # Age group
                "gender": 0.04,         # Gender
                "job_function": 0.04    # Professional role
            },
            "occasion": 0.10,           # Gift occasion (10%)
            "other": {                  # Other factors (10%)
                "color_theme": 0.05,    # Color preferences
                "misc": 0.05            
            }
        }
        
    def adjust_to_item_count(self, items: List[Dict], target_count: int, min_budget: float, max_budget: float) -> List[Dict]:
        """
        Adjust the selection to match the target item count while staying within budget.
        """
        if not items or target_count <= 0:
            return []
            
        current_count = len(items)
        
        # If we already have the right number, return as is
        if current_count == target_count:
            return items
            
        # Calculate current total cost
        current_total = sum(item.get('normalized_price', 0) for item in items if item.get('normalized_price') is not None)
        
        if current_count < target_count:
            # We need to add more items
            # Find items not already in our selection
            remaining_items = [item for item in self.items if item.get('normalized_price') is not None 
                              and item not in items]
                              
            # Sort by price (ascending) to add cheaper items first
            remaining_items.sort(key=lambda x: x.get('normalized_price', 0))
            
            for item in remaining_items:
                item_price = item.get('normalized_price', 0)
                if current_total + item_price <= max_budget:
                    items.append(item)
                    current_total += item_price
                    current_count += 1
                    
                    if current_count >= target_count:
                        break
        else:
            # We need to remove some items
            # Sort by price (ascending) so we remove cheaper items first
            # This helps maintain value while reducing count
            items.sort(key=lambda x: x.get('normalized_price', 0))
            
            while current_count > target_count:
                removed_item = items.pop(0)  # Remove the cheapest item
                current_total -= removed_item.get('normalized_price', 0)
                current_count -= 1
        
        return items
        
    def optimize_selection(self, items: List[Dict], min_budget: float, max_budget: float) -> List[Dict]:
        """
        Optimize the selection to maximize budget utilization while staying within limits.
        """
        if not items:
            return []
            
        # Calculate current total
        current_total = sum(item.get('normalized_price', 0) for item in items if item.get('normalized_price') is not None)
        
        if current_total > max_budget:
            # Over budget, need to remove items
            # Sort by price (descending) to remove expensive items first
            items.sort(key=lambda x: x.get('normalized_price', 0), reverse=True)
            
            while items and current_total > max_budget:
                removed_item = items.pop(0)  # Remove the most expensive item
                current_total -= removed_item.get('normalized_price', 0)
        
        elif current_total < min_budget:
            # Under minimum budget, try to add more items
            remaining_items = [item for item in self.items if item.get('normalized_price') is not None 
                              and item not in items]
                              
            # Sort by price (descending) to add valuable items first
            remaining_items.sort(key=lambda x: x.get('normalized_price', 0), reverse=True)
            
            for item in remaining_items:
                item_price = item.get('normalized_price', 0)
                if current_total + item_price <= max_budget:
                    items.append(item)
                    current_total += item_price
                    
                    if current_total >= min_budget:
                        break
        
        return items
    
    def select_items(self, criteria: str, min_budget: float, max_budget: float, item_count: int = None,
                    age_group: str = "", gift_occasion: str = "", color_theme: str = "",
                    job_function: str = "", gender: str = "", quantity: str = "",
                    total_budget: float = 500.0) -> List[Dict]:  # Added total_budget parameter with default
        """
        Use OpenAI to select items based on the criteria with weighted importance.
        Returns a list of items that fit the criteria.
        """
        # Prepare the data for OpenAI (limit the number of items to avoid token limits)
        items_sample = self.items[:50]  # Take a sample to avoid token limits
        
        # Extract discount information for each item
        for item in items_sample:
            has_discount, discount_info, _ = extract_discount_info(item)
            if has_discount:
                item['has_bulk_discount'] = True
                item['discount_info'] = discount_info
        
        items_data = json.dumps([{
            "name": item['name'],
            "type": item['type'],
            "price": item.get('normalized_price'),
            "description": item.get('short_description', '')[:100],
            "labels": item.get('labels', []),
            "has_bulk_discount": item.get('has_bulk_discount', False),
            "discount_info": item.get('discount_info', '')
        } for item in items_sample])
        
        # Create the system prompt with weighted criteria
        system_prompt = """
        You are a gift selection expert. Your task is to select items that best match the user's criteria with the following priority weights:

        1. QUERY (30%): The user's free text description of what they're looking for is the most important factor.
           - Pay close attention to specific item types, materials, or features mentioned
           - Understand the intent and purpose behind the request

        2. BUDGET CONSIDERATIONS (30%):
           - Budget range per package (15%): Select items that fit within the specified budget range
           - Total budget (10%): Consider how many packages can be created within the total budget
           - Number of items per package (5%): Aim to include the requested number of items

        3. DEMOGRAPHICS (20%):
           - Age Group (4%): Select age-appropriate items
           - Gender (8%): Consider gender preferences if specified
           - Job Function/Professional role (8%): Select items appropriate for the recipient's professional context

        4. OCCASION (10%):
           - Match items to the specific occasion or purpose of the gift

        5. OTHER FACTORS (10%):
           - Color theme (5%): Consider color preferences

        Pay special attention to items that offer bulk discounts and prioritize them when appropriate.
        Return your selections as a JSON array of item names that meet the criteria.
        """
        
        if item_count:
            system_prompt += f"\nSelect EXACTLY {item_count} items for the package if possible."
        else:
            system_prompt += "\nTry to select MORE THAN ONE item."
            
        budget_text = f"a maximum budget of S${max_budget:.2f}"
        if min_budget > 0:
            budget_text = f"a budget range of S${min_budget:.2f} to S${max_budget:.2f}"
        
        item_count_text = ""
        if item_count:
            item_count_text = f" containing exactly {item_count} items"
        
        # Build a structured user prompt that clearly separates the different criteria by importance
        user_prompt = f"""
        I have {budget_text} and I'm looking for items{item_count_text} that match these criteria:

        PRIMARY REQUIREMENT (30% weight):
        {criteria}

        BUDGET DETAILS (30% weight):
        - Budget per package: {budget_text}
        - Total budget available: S${total_budget:.2f}
        - Desired items per package: {item_count if item_count else "Multiple items"}

        RECIPIENT DEMOGRAPHICS (20% weight):
        """
        
        # Add demographic information if provided
        if age_group and age_group != "Choose an option":
            user_prompt += f"- Age group: {age_group}\n        "
        if gender and gender != "Choose an option" and gender != "does not really matter":
            user_prompt += f"- Gender: {gender}\n        "
        if job_function and job_function != "Choose an option":
            user_prompt += f"- Job function: {job_function}\n        "
        
        user_prompt += f"""
        OCCASION (10% weight):
        {gift_occasion if gift_occasion and gift_occasion != "Choose an option" else "Not specified"}

        OTHER PREFERENCES (10% weight):
        """
        
        # Add other preferences if provided
        if color_theme and color_theme != "Choose an option":
            user_prompt += f"- Color preference: {color_theme}\n        "
        if quantity and quantity != "Choose quantity":
            user_prompt += f"- Quantity: {quantity}\n        "
            
        user_prompt += f"""
        Here are the available items:
        {items_data}
        
        Please select the items that best match my criteria according to the weighted priorities and MAXIMIZE the budget utilization while staying under the maximum budget limit. Try to get as close as possible to the maximum budget.
        
        If an item has bulk discounts available (has_bulk_discount=true), prioritize these items when appropriate for the budget.
        
        Return a JSON object with a key called "items" that contains an array of item names, like this:
        {{"items": ["Item 1", "Item 2", "Item 3"]}}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            # Extract the selected item names
            result = json.loads(response.choices[0].message.content)
            selected_item_names = result.get("items", [])
            if not isinstance(selected_item_names, list):
                # Try to handle different response formats
                if isinstance(result, list):
                    selected_item_names = result
                else:
                    # Look for any list in the response
                    for value in result.values():
                        if isinstance(value, list):
                            selected_item_names = value
                            break
            
            # Find the corresponding items from our pool
            selected_items = []
            for name in selected_item_names:
                for item in self.items:
                    if name.lower() in item['name'].lower() or item['name'].lower() in name.lower():
                        # Add discount info to the item
                        has_discount, discount_info, formatted_discount = extract_discount_info(item)
                        if has_discount:
                            item['has_bulk_discount'] = True
                            item['discount_info'] = discount_info
                            item['formatted_discount'] = formatted_discount
                        
                        selected_items.append(item)
                        break
            
            # Check if we need to adjust the selection based on the exact item count
            if item_count and len(selected_items) != item_count:
                selected_items = self.adjust_to_item_count(selected_items, item_count, min_budget, max_budget)
            else:
                # Optimize selection to maximize budget utilization
                selected_items = self.optimize_selection(selected_items, min_budget, max_budget)
            
            return selected_items
        except Exception as e:
            print(f"Error calling OpenAI API: {str(e)}")
            return []

# Now we need to update the process_query method in GiftBundleChatbot to pass the total_budget parameter
class GiftBundleChatbot:
    def __init__(self, items):
        self.items = items
        self.budget_agent = BudgetAgent(items)
        self.selection_agent = SelectionAgent(items)
    
    def process_query(self, query: str, min_budget: float = 0, max_budget: float = 500,
                     total_budget: float = 500, item_count: int = None, age_group: str = "",
                     gift_occasion: str = "", color_theme: str = "", job_function: str = "",
                     gender: str = "", quantity: str = "") -> Tuple[str, List[Dict]]:
        """
        Process a user query and return recommendations based on weighted criteria.
        """
        # Use the selection agent to get items based on criteria with weights
        selected_items = self.selection_agent.select_items(
            criteria=query or "Find me gift items",
            min_budget=min_budget,
            max_budget=max_budget,
            item_count=item_count,
            age_group=age_group,
            gift_occasion=gift_occasion,
            color_theme=color_theme,
            job_function=job_function,
            gender=gender,
            quantity=quantity,
            total_budget=total_budget  # Pass the total_budget parameter
        )
        
        if not selected_items:
            # Fallback to the budget agent for optimization
            selected_items = self.budget_agent.optimize_bundle(min_budget, max_budget, self.items, query)
            
            # Adjust to match the requested item count if specified
            if item_count and len(selected_items) != item_count:
                # Use the selection agent's method to adjust the count
                selected_items = self.selection_agent.adjust_to_item_count(
                    selected_items, item_count, min_budget, max_budget
                )
        
        # Check if the selected items fit within the per-package budget range
        fits_budget, total_cost, explanation = self.budget_agent.calculate_bundle(min_budget, max_budget, selected_items)
        
        # Calculate how many packages fit within the total budget
        num_packages, total_packages_cost, total_budget_explanation = self.budget_agent.adjust_bundle_to_fit_total_budget(
            selected_items, min_budget, max_budget, total_budget
        )
        
        # Format the response
        if not selected_items:
            return "I couldn't find any items matching your criteria within your budget range. Please try different criteria or adjust your budget.", []
        
        budget_text = f"per-package budget of S${min_budget:.2f}-S${max_budget:.2f}"
        total_budget_text = f"total budget of S${total_budget:.2f}"
        item_count_text = f"with {len(selected_items)} items per package" if item_count else ""
                
        response = f"Based on your criteria: '{query or 'Gift items'}'"
        
        # Add filter information to response if selected
        filters_used = []
        if item_count:
            filters_used.append(f"Items per package: {item_count}")
        if age_group and age_group != "Choose an option":
            filters_used.append(f"Age group: {age_group}")
        if gift_occasion and gift_occasion != "Choose an option":
            filters_used.append(f"Occasion: {gift_occasion}")
        if color_theme and color_theme != "Choose an option":
            filters_used.append(f"Color: {color_theme}")
        if job_function and job_function != "Choose an option":
            filters_used.append(f"Job function: {job_function}")
        if gender and gender != "Choose an option" and gender != "does not really matter":
            filters_used.append(f"Gender: {gender}")
        if quantity and quantity != "Choose quantity":
            filters_used.append(f"Quantity: {quantity}")
        if filters_used:
            response += f"\nFilters: {', '.join(filters_used)}"
            
        response += f"\nBudget: {budget_text} {item_count_text} with {total_budget_text}\n\nI recommend:\n\n"
        
        # Check for items with bulk discounts to highlight them
        discount_items = []
        
        for item in selected_items:
            price_display = f"S${item['normalized_price']:.2f}" if item['normalized_price'] is not None else "Price not available"
            
            # Check if this item has bulk discount information
            if item.get('has_bulk_discount', False):
                response += f"- {item['name']} ({price_display}) 💰 BULK DISCOUNT AVAILABLE 💰\n  {item.get('short_description', 'No description')[:100]}...\n"
                response += f"  {item.get('formatted_discount', '')}\n\n"
                discount_items.append(item['name'])
            else:
                response += f"- {item['name']} ({price_display})\n  {item.get('short_description', 'No description')[:100]}...\n\n"
        
        response += f"\n{explanation.replace('$', 'S$')}"
        
        # Add total budget explanation
        response += f"\n\n{total_budget_explanation.replace('$', 'S$')}"
        
        # Add special note about bulk discounts if any were found
        if discount_items:
            response += f"\n\n💰 SPECIAL NOTE: {len(discount_items)} item(s) in your selection offer bulk discounts: {', '.join(discount_items)}. Consider ordering in larger quantities to save money!"
        
        # Add recommendation to adjust budget if needed
        if not fits_budget:
            if total_cost > max_budget:
                response += "\n\nWould you like to increase your maximum per-package budget or see a different selection?"
            elif min_budget > 0 and total_cost < min_budget:
                response += "\n\nWould you like to decrease your minimum per-package budget or see a selection with additional items?"
        
        if num_packages == 0:
            response += "\n\nThe cost of this bundle exceeds your total budget. Would you like to increase your total budget or see a more affordable selection?"
        
        return response, selected_items
        
def parse_budget_range(budget_range):
    """Parse a budget range string into min and max values"""
    if budget_range == "Below S$10":
        return 0, 10
    elif budget_range == "S$10 to S$20":
        return 10, 20
    elif budget_range == "S$20 to S$35":
        return 20, 35
    elif budget_range == "S$35 to S$55":
        return 35, 55
    elif budget_range == "S$55 to S$80":
        return 55, 80
    else:
        # Default range if no match
        return 0, 500

def extract_discount_info(item):
    """
    Extract bulk discount information from item description.
    Returns: (has_discount, discount_info, formatted_info)
    """
    has_discount = False
    discount_info = None
    formatted_info = ""
    
    # Check if the item has a description
    description = item.get('short_description', '') or item.get('description', '')
    if not description:
        return has_discount, discount_info, formatted_info
    
    # Keywords that might indicate a bulk discount
    discount_keywords = [
        'bulk discount', 'volume discount', 'quantity discount',
        'bulk pricing', 'buy more save more', 'discount for quantities',
        'bulk purchase', 'special pricing', 'wholesale price',
        'bulk orders', 'quantity pricing', 'discount for bulk'
    ]
    
    description_lower = description.lower()
    
    # Check for discount keywords
    for keyword in discount_keywords:
        if keyword in description_lower:
            has_discount = True
            break
    
    if has_discount:
        # Try to extract sentences containing discount information
        sentences = description.split('.')
        discount_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            sentence_lower = sentence.lower()
            
            for keyword in discount_keywords:
                if keyword in sentence_lower and sentence:
                    discount_sentences.append(sentence)
                    break
        
        if discount_sentences:
            discount_info = '. '.join(discount_sentences) + '.'
            formatted_info = f"<strong>Bulk Discount:</strong> {discount_info}"
        else:
            # If we can't extract specific sentences, use entire description
            discount_info = description
            formatted_info = f"<strong>Bulk Discount Available</strong> (see description for details)"
    
    return has_discount, discount_info, formatted_info

# Set up the Gradio interface
def gift_finder_interface(budget_range, budget_total, package_item_count, color, query):
    """
    Fixed gift_finder_interface function that incorporates package item count and discount information
    """
    # Parse the budget range
    min_budget, max_budget = parse_budget_range(budget_range)
    
    # Get the total budget
    try:
        if budget_total and float(budget_total) > 0:
            total_budget = float(budget_total)
        else:
            total_budget = 100.0
    except (ValueError, TypeError):
        total_budget = 100.0
    # Get the number of items per package
    item_count = int(package_item_count) if package_item_count else None
    
    # Initialize the chatbot
    chatbot = GiftBundleChatbot(all_items)
    
    # Process the query with all filters
    response, selected_items = chatbot.process_query(
        query=query,
        min_budget=min_budget,
        max_budget=max_budget,
        total_budget=total_budget,
        item_count=item_count,
        color_theme=color,
    )
    
    # Create DataFrame for bundle display
    if selected_items:
        # Calculate bundle cost
        package_cost = sum(item['normalized_price'] for item in selected_items if item['normalized_price'] is not None)
        
        # Calculate how many packages fit in total budget
        max_packages = int(total_budget / package_cost) if package_cost > 0 else 0
        total_cost = package_cost * max_packages if max_packages > 0 else 0
        
        # Create dataframe with discount information highlighted
        bundle_data = []
        for item in selected_items:
            price_display = f"S${item['normalized_price']:.2f}" if item['normalized_price'] is not None else "N/A"
            
            # Add discount note if available
            discount_note = ""
            if item.get('has_bulk_discount', False):
                discount_note = "💰 BULK DISCOUNT AVAILABLE"
            
            # Prepare description with discount info if available
            description = item.get('short_description', 'No description')[:100]
            if item.get('formatted_discount', ''):
                description += f"\n{item.get('formatted_discount', '')}"
            
            bundle_data.append({
                "Name": item['name'],
                "Price (S$)": price_display,
                "Type": item['type'],
                "Bulk Discount": discount_note,
                "Description": description
            })
        
        bundle_df = pd.DataFrame(bundle_data)
        
        budget_utilization = (total_cost / total_budget) * 100 if total_budget > 0 else 0
        
        bundle_summary = f"Package Cost: S${package_cost:.2f}\n" \
                         f"Items per Package: {len(selected_items)}\n" \
                         f"Number of Packages Possible: {max_packages}\n" \
                         f"Total Cost: S${total_cost:.2f}\n" \
                         f"Total Budget: S${total_budget:.2f}\n" \
                         f"Budget Utilization: {budget_utilization:.1f}%"
        
        # Create HTML for displaying images with discount badges
        html_content = "<div style='display: flex; flex-wrap: wrap; gap: 20px;'>"
        count = 0

        # Debug: Print image information for each item
        print("Image debug information:")
        for i, item in enumerate(selected_items):
            print(f"Item {i+1}: {item['name']}")
            print(f"  Has 'images' key: {'Yes' if 'images' in item else 'No'}")
            if 'images' in item:
                print(f"  Images value type: {type(item['images'])}")
                print(f"  Images value: {str(item['images'])[:100]}")  # Show first 100 chars

        # Process each item for images
        for item in selected_items:
            # Check if the item has images key
            if 'images' in item and item['images']:
                try:
                    # Get the image URL - handle different possible formats
                    image_url = None
                    
                    if isinstance(item['images'], str):
                        # Direct URL string
                        image_url = item['images']
                    elif isinstance(item['images'], list) and len(item['images']) > 0:
                        # List of URLs - take the first one
                        image_url = item['images'][0]
                    elif isinstance(item['images'], dict) and len(item['images']) > 0:
                        # Dictionary of URLs - take the first value
                        image_url = list(item['images'].values())[0]
                    
                    # If it's a relative URL, convert to absolute (example)
                    if image_url and image_url.startswith('/'):
                        # This is just an example - update with your actual domain
                        image_url = f"https://yourdomain.com{image_url}"
                    
                    # Print the processed URL for debugging
                    print(f"Processed image URL for {item['name']}: {image_url}")
                    
                    if image_url:
                        # Create HTML for the image with caption
                        item_price = f"S${item['normalized_price']:.2f}" if item['normalized_price'] is not None else "N/A"
                        
                        # Add a discount badge if available
                        discount_badge = ""
                        if item.get('has_bulk_discount', False):
                            discount_badge = """
                            <div style='position: absolute; top: 5px; right: 5px; background-color: #FF9800; color: white; padding: 5px; border-radius: 4px; font-size: 0.8em;'>
                                BULK DISCOUNT
                            </div>
                            """
                        
                        html_content += f"""
                        <div style='border: 1px solid #ddd; border-radius: 8px; padding: 10px; max-width: 250px; position: relative;'>
                            {discount_badge}
                            <img src="{image_url}" alt="{item['name']}" style='width: 100%; max-height: 200px; object-fit: contain;'>
                            <p style='margin-top: 8px; font-weight: bold;'>{item['name']}</p>
                            <p>{item_price}</p>
                        """
                        
                        # Add discount info if available
                        if item.get('formatted_discount', ''):
                            html_content += f"""<p style='color: #FF9800; font-weight: bold;'>{item.get('formatted_discount', '')}</p>
                            """
                        
                        html_content += "</div>"
                        count += 1
                except Exception as e:
                    print(f"Error processing image for {item['name']}: {str(e)}")

        # Close the container div
        html_content += "</div>"

        # If no images were found
        if count == 0:
            html_content = """
            <div>
                <p>No images were found for the selected items.</p>
                <p>Check the console logs for debugging information about the image URLs.</p>
            </div>
            """
    else:
        bundle_df = pd.DataFrame(columns=["Name", "Price (S$)", "Type", "Bulk Discount", "Description"])
        bundle_summary = "No items selected"
        html_content = "<p>No items selected.</p>"
    
    return response, bundle_df, bundle_summary, html_content

# Custom CSS to match the Gift Market homepage style
css = """
:root {
    --primary-color: #87CEEB;
    --secondary-color: #3C3B6E;
    --background-color: #f0f2f5;
    --border-color: #ddd;
}

body {
    font-family: 'Arial', sans-serif;
    background-color: var(--background-color);
}

.main-container {
    max-width: 1200px;
    margin: 0 auto;
}

.header {
    background-color: white;
    padding: 10px 0;
    border-bottom: 1px solid var(--border-color);
}

h1.title {
    color: var(--primary-color);
    font-weight: bold;
    font-size: 2.5em;
    margin: 0;
    padding: 10px 0;
}

.section-header {
    background-color: var(--background-color);
    padding: 8px;
    margin-top: 10px;
    border-radius: 5px;
    font-size: 1.2em;
    color: #333;
    font-weight: bold;
}

.section-number {
    display: inline-block;
    width: 24px;
    height: 24px;
    background-color: var(--secondary-color);
    color: white;
    border-radius: 50%;
    text-align: center;
    margin-right: 10px;
}

.btn-primary {
    background-color: var(--primary-color);
    border-color: var(--primary-color);
}

.btn-primary:hover {
    background-color: #8f1c2a;
    border-color: #8f1c2a;
}

.filter-row {
    display: flex;
    flex-direction: row;
    flex-wrap: nowrap;
    gap: 10px;
    margin-bottom: 8px;
    padding: 6px;
    background-color: white;
    border-radius: 5px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

/* Specific styling for columns in filter rows */
.filter-row .gr-column {
    flex: 1;
    min-width: 250px;
}

.budget-btn {
    background-color: white;
    border: 1px solid var(--border-color);
    color: #333;
    padding: 8px 15px;
    border-radius: 20px;
    margin: 5px;
    cursor: pointer;
    transition: all 0.2s;
}

.budget-btn:hover, .budget-btn.active {
    background-color: var(--primary-color);
    color: white;
    border-color: var(--primary-color);
}

.results-container {
    background-color: white;
    border-radius: 5px;
    padding: 15px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.footer {
    text-align: center;
    padding: 20px 0;
    margin-top: 30px;
    font-size: 0.9em;
    color: #87CEEB;
}

/* Style for checkboxes */
.checkbox-container {
    display: flex;
    gap: 15px;
    margin: 10px 0;
}

/* Style for the search box */
.search-box {
    display: flex;
    margin: 15px 0;
}

.search-box input {
    flex-grow: 1;
    padding: 8px 15px;
    border: 1px solid var(--border-color);
    border-radius: 4px 0 0 4px;
}

.search-box button {
    background-color: var(--secondary-color);
    color: white;
    border: none;
    padding: 8px 15px;
    border-radius: 0 4px 4px 0;
    cursor: pointer;
}

.gr-dropdown {
    min-height: 35px !important;
}

.gr-dropdown select {
    height: 35px !important;
    padding: 5px !important;
}

.gr-box {
    min-height: 35px !important;
}

.gr-radio-group {
    gap: 5px !important;
    margin: 2px 0 !important;
}

select {
    height: 35px !important;
    padding: 5px !important;
}

/* Responsive design for smaller screens */
@media (max-width: 768px) {
    .filter-row {
        flex-wrap: wrap;
    }
    .filter-row .gr-column {
        min-width: 100%;
    }
}
"""

# Define the Gradio interface
with gr.Blocks(css=css, title="Gift Finder") as demo:
    #Header
    # Define styles separately for better organization
    header_styles = {
        "container": """
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 20px;
            width: 100%;
            flex-wrap: wrap;
            gap: 20px;
        """,
        "logo_section": """
            display: flex;
            align-items: center;
            gap: 15px;
        """,
        "logo": """
            height: 50px;
            width: auto;
            object-fit: contain;
        """,
        "title": """
            color: #87CEEB;
            font-weight: bold;
            margin: 0;
            font-size: clamp(1.5rem, 2vw, 2rem);
        """,
        "nav": """
            display: flex;
            gap: 20px;
            align-items: center;
        """,
        "nav_item": """
            display: flex;
            align-items: center;
            cursor: pointer;
        """
    }

    with gr.Row(elem_classes=["header"]):
        gr.HTML(f"""
            <div style="{header_styles['container']}">
                <div style="{header_styles['logo_section']}">
                    <img src="data:image/png;base64,{logo_base64}" 
                         alt="PrintNGift Logo" 
                         style="{header_styles['logo']}">
                    <div>
                        <h1 style="{header_styles['title']}">
                            Your Gift Finder
                        </h1>
                    </div>
                </div>
                <nav style="{header_styles['nav']}">
                    <div style="{header_styles['nav_item']}">
                        <span style="font-weight: bold; margin-right: 10px;">Shop</span>
                        <span style="font-size: 0.8rem;">▼</span>
                    </div>
                    <div style="{header_styles['nav_item']}">
                        <span style="font-weight: bold;">My Enquiry (0)</span>
                    </div>
                </nav>
            </div>
        """)
    
    # Main title
    # Search bar
    gr.HTML("""
        <div class="section-header">
            <span class="section-number">0</span> Describe what you're looking for*
        </div>
    """)

    with gr.Row(elem_classes=["search-box"]):
        query = gr.Textbox(
            placeholder="Example: Find me office supplies or I need premium drinkware items",
            label="Requirements"
        )
    
    # Budget section
    gr.HTML("""
        <div class="section-header">
            <span class="section-number">1</span>Budget (S$): per package + total budget + items in bundle*
        </div>
    """)

    # Budget buttons and input
    with gr.Row(elem_classes=["filter-row"]):
        with gr.Column(scale=20):
            package_item_count = gr.Slider(
                minimum=1,
                maximum=7,
                value=3,
                step=1,
                label="Number",
                info="Items/gift package"
            )
        with gr.Column(scale=60):
            budget_range = gr.Radio(
                choices=BUDGET_RANGES,
                label="Budget Per Bundle Package",
                value=BUDGET_RANGES[0]
            )
        with gr.Column(scale=20):
            budget_total = gr.Number(
                label="Total Budget (S$)",
                minimum=10,
                value="100",
                info="Net of package cost"
            )

    # Combined row for Age Group, Gender, and Colour Theme
    with gr.Row(elem_classes=["filter-row"]):
        with gr.Column(scale=2):
            gr.HTML("""
                <div class="section-header" style="margin-top: 0;">
                    <span class="section-number">2</span> Target Age Group*
                </div>
            """)
            age_group = gr.Dropdown(
                choices=["Choose an option", "<18", "18-30", "30-40", "40-50", "50-60", ">60"],
                label="Age",
                value="Choose an option"
            )
        
        with gr.Column(scale=2):
            gr.HTML("""
                <div class="section-header" style="margin-top: 0;">
                    <span class="section-number">3</span> Target Gender
                </div>
            """)
            gender = gr.Dropdown(
                choices=["Choose an option", "Male", "Female", "does not really matter"],
                label="Gender",
                value="Choose an option"
            )
        
        with gr.Column(scale=2):
            gr.HTML("""
                <div class="section-header" style="margin-top: 0;">
                    <span class="section-number">4</span> Colour Theme
                </div>
            """)
            color_theme = gr.Dropdown(
                choices=[
                    "Choose an option",
                    "Black", "White", "Off-White", "Brown", "Red", "Blue", "Gray",
                    "Gold", "Yellow", "Purple", "Pink", "Green", "Silver",
                    "Orange", "Multi-color", "Transparent"
                ],
                label="Color",
                value="Choose an option"
            )
    # Gift Occasion section
    gr.HTML("""
        <div class="section-header">
            <span class="section-number">5</span> Gift Occasion
        </div>
    """)

    with gr.Row(elem_classes=["filter-row"]):
        gift_occasion = gr.Dropdown(
            choices=[
                "Choose an option",
                "Festive Celebration",
                "Long Service Award",
                "Corporate Milestones",
                "Onboarding",
                "Christmas/Year-End Celebration",
                "Annual Dinner & Dance",
                "All The Best!",
                "Others"
            ],
            label="Occasion",
            value="Choose an option",
            elem_classes=["gr-dropdown"]
        )

    # Job Function section
    gr.HTML("""
        <div class="section-header">
            <span class="section-number">6</span> Recipient's Job Function
        </div>
    """)

    with gr.Row(elem_classes=["filter-row"]):
        job_function = gr.Dropdown(
            choices=[
                "Choose an option",
                "C-Suite",
                "Sales & Business Development",
                "Finance",
                "Operations",
                "Human Resource",
                "Engineering",
                "Information Technology",
                "Marketing & Communications",
                "Others"
            ],
            label="Recipient",
            value="Choose an option",
            elem_classes=["gr-dropdown"]
        )

    # Results tabs
    with gr.Tabs():
        with gr.TabItem("Recommendations"):
            response = gr.Textbox(label="Recommendation Details", lines=15)
        with gr.TabItem("Bundle Summary"):
            bundle_summary = gr.Textbox(label="Bundle Statistics", lines=3)
            bundle_table = gr.DataFrame(label="Selected Items")
        with gr.TabItem("Bundle Pictures"):
            bundle_images = gr.HTML(label="Product Images")
    
    # Function to determine the final budget range
    def get_final_budget_range(range1, range2):
        return range1 if range1 else range2
    
    def modified_interface(budget_range1, budget_total, package_item_count, age_group, gift_occasion, color_theme, job_function, gender, query):
        """
        Updated interface function to handle per-package budget range, total budget,
        specific item count per package, and highlight bulk discounts
        """
    # Get the budget range for individual items in the package
        budget_range = budget_range1 if budget_range1 else "Below S$10"  # Default if nothing selected

        # Parse the budget range for individual items
        min_budget, max_budget = parse_budget_range(budget_range)

        # Get the total budget - ensure it's properly converted to float
        try:
            total_budget = float(budget_total) if budget_total else 100.0
        except (ValueError, TypeError):
            total_budget = 100.0

        # Get the number of items per package - ensure it's properly converted to int
        try:
            item_count = int(package_item_count) if package_item_count else None
        except (ValueError, TypeError):
            item_count = None
        # Get the number of items per package
        item_count = int(package_item_count) if package_item_count else None
        
        # Initialize the chatbot
        chatbot = GiftBundleChatbot(all_items)
        
        # Process the query with all filters
        response, selected_items = chatbot.process_query(
            query=query,
            min_budget=min_budget,
            max_budget=max_budget,
            total_budget=total_budget,
            item_count=item_count,
            age_group=age_group,
            gift_occasion=gift_occasion,
            color_theme=color_theme,
            job_function=job_function,
            gender=gender
        )

        # Create DataFrame for bundle display
        if selected_items:
            # Calculate the per-package cost
            package_cost = sum(item['normalized_price'] for item in selected_items if item['normalized_price'] is not None)
            
            # Calculate max number of packages that fit within total budget
            max_packages = int(total_budget / package_cost) if package_cost > 0 else 0
            total_cost = package_cost * max_packages if max_packages > 0 else 0
            
            # Create dataframe with discount information highlighted
            bundle_data = []
            for item in selected_items:
                price_display = f"S${item['normalized_price']:.2f}" if item['normalized_price'] is not None else "N/A"
                
                # Add discount note if available
                discount_note = ""
                if item.get('has_bulk_discount', False):
                    discount_note = "💰 BULK DISCOUNT AVAILABLE"
                
                # Prepare description with discount info if available
                description = item.get('short_description', 'No description')[:100]
                if item.get('formatted_discount', ''):
                    description += f"\n{item.get('formatted_discount', '')}"
                
                bundle_data.append({
                    "Name": item['name'],
                    "Price (S$)": price_display,
                    "Type": item['type'],
                    "Bulk Discount": discount_note,
                    "Description": description
                })
            
            bundle_df = pd.DataFrame(bundle_data)
            
            budget_utilization = (total_cost / total_budget) * 100 if total_budget > 0 else 0
            
            bundle_summary = f"Package Cost: S${package_cost:.2f}\n" \
                            f"Items per Package: {len(selected_items)}\n" \
                            f"Number of Packages Possible: {max_packages}\n" \
                            f"Total Cost: S${total_cost:.2f}\n" \
                            f"Total Budget: S${total_budget:.2f}\n" \
                            f"Budget Utilization: {budget_utilization:.1f}%"
            
            # Create HTML for displaying images with discount badges
            html_content = "<div style='display: flex; flex-wrap: wrap; gap: 20px;'>"
            count = 0

            # Debug: Print image information for each item
            print("Image debug information:")
            for i, item in enumerate(selected_items):
                print(f"Item {i+1}: {item['name']}")
                print(f"  Has 'images' key: {'Yes' if 'images' in item else 'No'}")
                if 'images' in item:
                    print(f"  Images value type: {type(item['images'])}")
                    print(f"  Images value: {str(item['images'])[:100]}")  # Show first 100 chars

            for item in selected_items:
                # Check if the item has images key
                if 'images' in item and item['images']:
                    try:
                        # Get the image URL - handle different possible formats
                        image_url = None
                        
                        if isinstance(item['images'], str):
                            # Direct URL string
                            image_url = item['images']
                        elif isinstance(item['images'], list) and len(item['images']) > 0:
                            # List of URLs - take the first one
                            image_url = item['images'][0]
                        elif isinstance(item['images'], dict) and len(item['images']) > 0:
                            # Dictionary of URLs - take the first value
                            image_url = list(item['images'].values())[0]
                        
                        # If it's a relative URL, convert to absolute (example)
                        if image_url and image_url.startswith('/'):
                            # This is just an example - update with your actual domain
                            image_url = f"https://yourdomain.com{image_url}"
                        
                        # Print the processed URL for debugging
                        print(f"Processed image URL for {item['name']}: {image_url}")
                        
                        if image_url:
                            # Create HTML for the image with caption
                            item_price = f"S${item['normalized_price']:.2f}" if item['normalized_price'] is not None else "N/A"
                            
                            # Add a discount badge if available
                            discount_badge = ""
                            if item.get('has_bulk_discount', False):
                                discount_badge = """
                                <div style='position: absolute; top: 5px; right: 5px; background-color: #FF9800; color: white; padding: 5px; border-radius: 4px; font-size: 0.8em;'>
                                    BULK DISCOUNT
                                </div>
                                """
                            
                            html_content += f"""
                            <div style='border: 1px solid #ddd; border-radius: 8px; padding: 10px; max-width: 250px; position: relative;'>
                                {discount_badge}
                                <img src="{image_url}" alt="{item['name']}" style='width: 100%; max-height: 200px; object-fit: contain;'>
                                <p style='margin-top: 8px; font-weight: bold;'>{item['name']}</p>
                                <p>{item_price}</p>
                            """
                            
                            # Add discount info if available
                            if item.get('formatted_discount', ''):
                                html_content += f"""<p style='color: #FF9800; font-weight: bold;'>{item.get('formatted_discount', '')}</p>
                            """
                            
                            html_content += "</div>"
                            count += 1
                    except Exception as e:
                        print(f"Error processing image for {item['name']}: {str(e)}")

            # Close the container div
            html_content += "</div>"

            # If no images were found
            if count == 0:
                html_content = """
                <div>
                    <p>No images were found for the selected items.</p>
                    <p>Check the console logs for debugging information about the image URLs.</p>
                </div>
                """
        else:
            bundle_df = pd.DataFrame(columns=["Name", "Price (S$)", "Type", "Bulk Discount", "Description"])
            bundle_summary = "No items selected"
            html_content = "<p>No items selected.</p>"
        
        return response, bundle_df, bundle_summary, html_content
    
    search_btn = gr.Button("Get Recommendations (Before you press, check your inputs, no automatic input of budget for bundle package)", variant="primary")
    # Search button click handler
    
    search_btn.click(
        fn=modified_interface,
        inputs=[budget_range, budget_total, package_item_count, age_group, gift_occasion, color_theme, job_function, gender, query],
        outputs=[response, bundle_table, bundle_summary, bundle_images]
    )
    # You can also update the examples to include total_budget values
    gr.Examples(
        examples=[
            ["S$10 to S$20", 100, "30-40", "Corporate Milestones", "Black", "C-Suite", "Male", "I need some premium wareable items"],
            ["S$35 to S$55", 200, "30-40", "Festive Celebration", "Blue", "Marketing & Communications", "Female", "Find me clothing items suitable for corporate events"],
            ["S$20 to S$35", 150, "18-30", "Long Service Award", "Silver", "Information Technology", "does not really matter", "Recommend tech gadgets"],
            ["S$55 to S$80", 300, "40-50", "All The Best!", "Multi-color", "Operations", "does not really matter", "I need some travel essentials"]
        ],
        inputs=[budget_range, budget_total, age_group, gift_occasion, color_theme, job_function, gender, query]
    )

    # Footer
    gr.HTML("""
        <div class="footer">
            <p>© 2025 Gift Market. All rights reserved.</p>
        </div>
    """)
    

# Launch the app
if __name__ == "__main__":
    # Uncommment to load real data
    # all_items = load_sample_data()
    
    # Launch Gradio interface
    demo.launch()
