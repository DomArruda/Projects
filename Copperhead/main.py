"""
Copperhead Tutorial and Examples
A comprehensive guide to using Rust-inspired patterns in Python
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import time
import json
from decimal import Decimal

# Import everything from our Copperhead library
from copperhead import (
    Result, Option, rrange,  # Core types
    result as result_dec, option as option_dec,  # Decorators
)

@dataclass
class Order:
    id: int
    user_id: int
    items: List[Dict[str, float]]
    total: float
    shipping_address: Option[str]

def main():
    """
    Comprehensive demonstration of Copperhead patterns.
    Shows how to make Python code more robust using Rust-inspired patterns.
    """
    print("Copperhead - Bringing Rust's Safety to Python")
    print("=" * 50)

    #
    # 1. Basic Result Usage
    #
    print("\n1. Result Type - Better Error Handling")
    print("-" * 50)

    @result_dec
    def divide(a: float, b: float) -> float:
        return a / b

    # Multiple ways to handle Results
    calculations = [
        divide(10, 2),
        divide(10, 0),
        divide(15, 3)
    ]

    for calc in calculations:
        # Basic style
        if calc.is_ok():
            print(f"Success: {calc.unwrap()}")
        else:
            print(f"Error: {calc.unwrap_or('Unknown error')}")

        # Method style with error handling
        try:
            value = calc.unwrap()
            print(f"The value is: {value}")
        except Exception as e:
            print(f"Caught exception: {e}")

        # Transformation style
        result_str = calc.map(lambda x: f"The answer is {x}").unwrap_or_else(
            lambda e: f"Error occurred: {e}"
        )
        print(result_str)

    #
    # 2. Option Type - No More NoneType Errors
    #
    print("\n2. Option Type - Safe Null Handling")
    print("-" * 50)

    # Dictionary lookups
    users = {"alice": "Alice Smith", "bob": "Bob Jones"}

    @option_dec
    def find_user(username: str) -> Optional[str]:
        return users.get(username)

    # Demonstrate different Option handling patterns
    for username in ["alice", "carol"]:
        user = find_user(username)

        # Direct checking
        if user.is_some():
            print(f"Found user: {user.unwrap()}")
        else:
            print(f"No user: {username}")

        # Transform and provide default
        greeting = user.map(lambda name: f"Hello, {name}!").unwrap_or("Hello, stranger!")
        print(greeting)

    #
    # 3. Rust-style Ranges
    #
    print("\n3. Rust's Range Syntax")
    print("-" * 50)

    print("Different range styles:")
    # Basic ranges
    print("rrange['..5']:", list(rrange['..5']))          # [0, 1, 2, 3, 4]
    print("rrange['1..5']:", list(rrange['1..5']))        # [1, 2, 3, 4]
    print("rrange['1..=5']:", list(rrange['1..=5']))      # [1, 2, 3, 4, 5]

    # Infinite ranges (showing first few elements)
    infinite = iter(rrange['3..'])
    print("First 3 from infinite range:", [next(infinite) for _ in range(3)])  # [3, 4, 5]

    # Full range (showing first few elements)
    full_range = iter(rrange['..'])
    print("First 3 from full range:", [next(full_range) for _ in range(3)])   # [0, 1, 2]

    # Using ranges in a for loop
    print("\nCounting from 1 to 3:")
    for i in rrange['1..=3']:
        print(f"Count: {i}")

    #
    # 4. Practical Data Handling
    #
    print("\n4. Practical Data Processing")
    print("-" * 50)

    class OrderProcessor:
        @result_dec
        def process_order(self, data: Dict) -> Order:
            if not data.get('items'):
                raise ValueError("Order must contain items")

            total = sum(float(item.get('price', 0)) for item in data['items'])
            if total <= 0:
                raise ValueError("Order total must be positive")

            shipping_address = data.get('address')
            return Order(
                id=data.get('id', 0),
                user_id=data.get('user_id', 0),
                items=data['items'],
                total=total,
                shipping_address=Option.Some(shipping_address)
            )

    # Process some orders
    processor = OrderProcessor()
    orders_data = [
        {
            "id": 1,
            "user_id": 1,
            "items": [{"name": "Book", "price": 29.99}],
            "address": "123 Main St"
        },
        {
            "id": 2,
            "user_id": 2,
            "items": [],  # This will fail validation
            "address": None
        }
    ]

    for order_data in orders_data:
        result = processor.process_order(order_data)

        if result.is_ok():
            order = result.unwrap()
            print(f"Processed order {order.id}:")
            print(f"- Total: ${order.total:.2f}")
            print(f"- Shipping to: {order.shipping_address.unwrap_or('No address provided')}")
        else:
            print(f"Failed to process order: {result.unwrap_or_else(lambda e: str(e))}")

    #
    # 5. Advanced Combined Example
    #
    print("\n5. Advanced Combined Example")
    print("-" * 50)

    class UserStats:
        def __init__(self):
            self.orders: Dict[int, List[Order]] = {}

        @option_dec
        def get_user_orders(self, user_id: int) -> Optional[List[Order]]:
            return self.orders.get(user_id)

        @result_dec
        def calculate_user_metrics(self, user_id: int) -> Dict[str, float]:
            orders_option = self.get_user_orders(user_id)
            orders = orders_option.unwrap_or([])

            if not orders:
                raise ValueError(f"No orders found for user {user_id}")

            total_spent = sum(order.total for order in orders)
            average_order = total_spent / len(orders)

            return {
                'total_spent': total_spent,
                'average_order': average_order,
                'num_orders': len(orders)
            }

    # Demonstrate the full power of Copperhead patterns
    stats = UserStats()
    stats.orders = {
        1: [
            Order(1, 1, [{"name": "Book", "price": 29.99}], 29.99, Option.Some("123 Main St")),
            Order(2, 1, [{"name": "Laptop", "price": 999.99}], 999.99, Option.Some("123 Main St"))
        ]
    }

    # Process user metrics
    for user_id in [1, 2]:
        print(f"\nProcessing user {user_id}:")
        metrics_result = stats.calculate_user_metrics(user_id)

        # Transform and handle the result
        summary = metrics_result.map(lambda metrics: f"""
        Total Spent: ${metrics['total_spent']:.2f}
        Average Order: ${metrics['average_order']:.2f}
        Number of Orders: {metrics['num_orders']}
        """).unwrap_or_else(lambda e: f"Could not calculate metrics: {e}")

        print(summary)

if __name__ == "__main__":
    main()
