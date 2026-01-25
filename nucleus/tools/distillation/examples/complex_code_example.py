# Example: Complex Code (Entropic)
# This file demonstrates excessive complexity that should be REJECTED

def mega_complex_function(a, b, c, d, e, f, g):
    """Overly complex function with high McCabe complexity."""
    result = 0
    
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    if e > 0:
                        if f > 0:
                            if g > 0:
                                for i in range(a):
                                    for j in range(b):
                                        while i < c:
                                            if j % 2 == 0:
                                                if i % 3 == 0:
                                                    try:
                                                        result += i * j
                                                    except:
                                                        pass
                                            i += 1
    
    return result


class DeeplyNestedClass:
    """Class with excessive nesting depth."""
    
    class InnerClass1:
        class InnerClass2:
            class InnerClass3:
                class InnerClass4:
                    class InnerClass5:
                        def deep_method(self):
                            def inner_function():
                                def even_deeper():
                                    return "too deep"
                                return even_deeper()
                            return inner_function()


def god_function_that_does_everything():
    """
    This function is way too long (>50 lines).
    It does everything and has no clear single responsibility.
    """
    # Setup
    config = {}
    database = None
    cache = None
    
    # Database operations
    if database:
        database.connect()
        database.query("SELECT * FROM users")
        database.disconnect()
    
    # Cache operations
    if cache:
        cache.set("key", "value")
        cache.get("key")
        cache.delete("key")
    
    # Business logic 1
    for i in range(100):
        if i % 2 == 0:
            print(i)
    
    # Business logic 2
    for j in range(100):
        if j % 3 == 0:
            print(j)
    
    # Business logic 3
    for k in range(100):
        if k % 5 == 0:
            print(k)
    
    # Validation
    if config:
        validate_config(config)
    
    # Email sending
    send_email("admin@example.com", "Subject", "Body")
    
    # Logging
    log_to_file("operation.log", "Everything done")
    
    return True


def validate_config(config):
    pass

def send_email(to, subject, body):
    pass

def log_to_file(filename, message):
    pass
