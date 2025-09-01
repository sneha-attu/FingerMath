"""
Safe expression evaluation using SymPy.
Handles mathematical expression parsing and calculation.
"""

from sympy import sympify, SympifyError
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import re

class ExpressionEvaluator:
    def __init__(self):
        self.current_expression = ""
        self.last_result = ""
        self.history = []
        
        # Define allowed symbols and functions for safety
        self.allowed_names = {
            'pi': 3.141592653589793,
            'e': 2.718281828459045,
        }
        
        # SymPy transformations for parsing
        self.transformations = (
            standard_transformations + 
            (implicit_multiplication_application,)
        )
    
    def add_token(self, token):
        """Add a token to the current expression."""
        if token in "0123456789":
            self.current_expression += token
        elif token in "+-*/":
            # Prevent consecutive operators
            if (self.current_expression and 
                self.current_expression[-1] not in "+-*/"):
                self.current_expression += token
        elif token == ".":
            # Add decimal point if valid
            if (self.current_expression and 
                self.current_expression[-1].isdigit() and
                "." not in self.current_expression.split("+-*/")[-1]):
                self.current_expression += token
    
    def backspace(self):
        """Remove the last character from expression."""
        if self.current_expression:
            self.current_expression = self.current_expression[:-1]
    
    def clear_expression(self):
        """Clear the current expression."""
        self.current_expression = ""
        self.last_result = ""
    
    def evaluate_expression(self):
        """Safely evaluate the current expression using SymPy."""
        if not self.current_expression:
            return "Error: Empty expression"
        
        try:
            # Clean the expression
            cleaned_expr = self._clean_expression(self.current_expression)
            
            if not cleaned_expr:
                return "Error: Invalid expression"
            
            # Parse and evaluate using SymPy
            parsed_expr = parse_expr(
                cleaned_expr,
                transformations=self.transformations,
                local_dict=self.allowed_names,
                evaluate=True
            )
            
            # Convert to float for display
            result = float(parsed_expr)
            
            # Format result
            if result.is_integer():
                formatted_result = str(int(result))
            else:
                formatted_result = f"{result:.6g}"
            
            # Store in history
            self.history.append({
                'expression': self.current_expression,
                'result': formatted_result
            })
            
            # Update state
            self.last_result = formatted_result
            self.current_expression = formatted_result  # Allow chaining operations
            
            return formatted_result
            
        except (SympifyError, ValueError, TypeError, ZeroDivisionError) as e:
            error_msg = f"Error: {str(e)}"
            self.last_result = error_msg
            return error_msg
        except Exception as e:
            error_msg = "Error: Invalid calculation"
            self.last_result = error_msg
            return error_msg
    
    def _clean_expression(self, expr):
        """Clean and validate the expression."""
        if not expr:
            return ""
        
        # Remove any non-mathematical characters
        cleaned = re.sub(r'[^0-9+\-*/.() ]', '', expr)
        
        # Remove trailing operators
        cleaned = re.sub(r'[+\-*/]+$', '', cleaned)
        
        # Check for basic validity
        if not cleaned or cleaned in "+-*/":
            return ""
        
        # Replace consecutive operators (except for negative numbers)
        cleaned = re.sub(r'[+\-*/]{2,}', lambda m: m.group()[-1], cleaned)
        
        return cleaned.strip()
    
    def get_current_expression(self):
        """Get the current expression string."""
        return self.current_expression
    
    def get_last_result(self):
        """Get the last calculation result."""
        return self.last_result
    
    def get_history(self):
        """Get calculation history."""
        return self.history.copy()
