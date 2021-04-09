def list_to_string(l):
   s = '['
   for x in l:
       s += f'{x:3.1f},'
   s = s[:-1] + ']'
   return s

def print_tensor(t):
    t = t.tolist()
    for l in t:
        print(list_to_string(l))

