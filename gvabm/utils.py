def can_pickle(obj):
    try:
        pickle.dumps(obj)
        return True
    except Exception as e:
        return False

def sanitize_object(obj):
    if can_pickle(obj):
        return obj
    if not hasattr(obj, '__dict__'):
        return None
    names = list(obj.__dict__.keys())
    for name in names:
        value = obj.__dict__[name]
        if not can_pickle(value):
            obj.__dict__[name] = sanitize_object(value)
    if can_pickle(obj):
        return obj
    else:
        print('Could not sanitize object:', obj)
        return None