It can be accessed as a module as `test.modules.op.*`


### Test target instantiation strategy: test-all, skip-some

Use `@skip` decorator to mark which torch.nn.Module to skip.

```py
# test.modules.op.add

class SimpleAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z = x + y
        z = z + x
        z = z + x
        z = z + z
        return (z,)

    def get_example_inputs(self):
        return (torch.ones(1), torch.ones(1))

@skip(reason="Too large!")
class VeryLargeSimpleAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z = x + y
        return (z,)

    def get_example_inputs(self):
        return (torch.ones(99999999), torch.ones(99999999))
```
