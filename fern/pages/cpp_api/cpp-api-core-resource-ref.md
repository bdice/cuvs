---
slug: api-reference/cpp-api-core-resource-ref
---

# Resource Ref

_Source header: `cuvs/core/resource_ref.hpp`_

## Types

<a id="device-resource-ref"></a>
### device_resource_ref

Stream-ordered reference to a device-accessible memory resource

```cpp
using device_resource_ref = cuda::mr::resource_ref<cuda::mr::device_accessible>;
```
