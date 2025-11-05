
def blog_authorship_labelgetter(age):
    if age < 18:
        return 0
    elif age < 28:
        return 1
    else:
        return 2
