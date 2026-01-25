# Example: Bad Code (Entropic)
# This file demonstrates anti-patterns and bloat that should be REJECTED

class AbstractFactoryFactory:
    """Over-engineered factory pattern - anti-pattern!"""
    
    def create_factory(self):
        return Factory()


class Factory:
    """Generic factory - code smell"""
    
    def create_object(self):
        return Object()


class Object:
    """Generic object - meaningless abstraction"""
    pass


class UserManager:
    """Manager anti-pattern - does everything"""
    
    def create_user(self, name):
        pass
    
    def delete_user(self, user_id):
        pass
    
    def update_user(self, user_id, data):
        pass
    
    def validate_user(self, user):
        pass
    
    def send_email_to_user(self, user_id):
        pass
    
    def log_user_activity(self, user_id):
        pass
    
    def generate_user_report(self, user_id):
        pass


class GodClass:
    """Knows and does too much - classic anti-pattern"""
    
    def __init__(self):
        self.database = None
        self.cache = None
        self.logger = None
        self.config = None
        self.email_service = None
        
    def do_everything(self, x, y, z):
        if x > 0:
            if y > 0:
                if z > 0:
                    for i in range(10):
                        for j in range(10):
                            for k in range(10):
                                if i % 2 == 0:
                                    if j % 2 == 0:
                                        if k % 2 == 0:
                                            # Deeply nested complexity
                                            pass
