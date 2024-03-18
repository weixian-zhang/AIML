def split_by_camel_case(content: str):
    result = []
    C = len(content) - 1
    i = C
    while i >= 0:
        if content[i].isupper() and i != C:
            result.append(content[i:])
            content = content[:i]
        i -= 1
    
    result.reverse()
    return ' '.join(result)


print(split_by_camel_case('AzureActiveDirectory'))
print(split_by_camel_case('SaaS'))
                    
