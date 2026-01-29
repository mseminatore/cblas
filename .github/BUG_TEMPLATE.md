# Bug Report

**Priority:** [Critical/High/Medium/Low]  
**Labels:** bug, [add relevant labels: correctness, memory-leak, crash, performance, etc.]  
**Status:** Open

---

## Description

[Clear and concise description of the bug]

---

## Environment

**Operating System:** [e.g., Windows 11, Ubuntu 22.04, macOS 14]  
**Compiler:** [e.g., GCC 12.2, Clang 15, MSVC 2022]  
**Architecture:** [e.g., x86_64, ARM64]  
**CBLAS Version:** [e.g., 0.16]  
**Build Configuration:**
- Multi-threading enabled: [Yes/No]
- SIMD enabled: [Yes/No]
- Thread count: [e.g., 4]

---

## Steps to Reproduce

1. [First step]
2. [Second step]
3. [And so on...]

**Minimal Code Example:**
```c
// Paste minimal reproducible code here
#include "cblas.h"

int main(void) {
    cblas_init(4);
    // Your test code
    cblas_shutdown();
    return 0;
}
```

---

## Expected Behavior

[What you expected to happen]

---

## Actual Behavior

[What actually happened]

**Error Messages/Output:**
```
[Paste any error messages, stack traces, or unexpected output]
```

---

## Investigation

**Root Cause (if known):**
[Analysis of what's causing the bug]

**Affected Code:**
- File: [e.g., dot.c]
- Function: [e.g., cblas_sdot()]
- Lines: [e.g., 145-152]

**Code Snippet:**
```c
// Paste relevant code showing the bug
for (CBLAS_INDEX i = 0; i < n; i++)
{
    sum += *x * *y;  // Issue here
    x += incx;
    y += incy;
}
```

---

## Proposed Fix

**Solution:**
[Describe the proposed fix]

**Implementation:**
```c
// Paste proposed code fix
for (CBLAS_INDEX i = 0; i < n; i++)
{
    sum += (*x) * (*y);  // Fixed with proper parentheses
    x += incx;
    y += incy;
}
```

**Alternative Approaches:**
- [Option 1]
- [Option 2]

---

## Impact Assessment

**Severity:** [Critical/High/Medium/Low]

**Affected Operations:**
- [e.g., cblas_sdot, cblas_ddot]
- [e.g., All Level-1 operations]

**Data Corruption Risk:** [Yes/No]  
**Crash Risk:** [Yes/No]  
**Performance Impact:** [Description]

**Workaround Available:** [Yes/No]  
**Workaround Description:**
[Describe temporary workaround if available]

---

## Testing

**Test Case:**
```c
// Paste test case that reproduces the bug
void test_bug_reproduction(void)
{
    float x[] = {1.0f, 2.0f, 3.0f};
    float y[] = {4.0f, 5.0f, 6.0f};
    float result = cblas_sdot(3, x, 1, y, 1);
    // Expected: 32.0, Actual: [incorrect value]
    assert(result == 32.0f);
}
```

**Verification Plan:**
- [ ] Unit test added/updated
- [ ] Test passes with fix applied
- [ ] No regression in existing tests
- [ ] Valgrind/sanitizer clean
- [ ] Verified on multiple platforms

---

## Additional Context

**Related Issues:**
- Issue #[number]
- PR #[number]

**Additional Information:**
[Any other context, screenshots, or relevant information]

**References:**
- [BLAS specification reference]
- [Related documentation]

---

## Acceptance Criteria

- [ ] Bug is reproducible with provided test case
- [ ] Root cause is identified and documented
- [ ] Fix is implemented and tested
- [ ] No performance regression
- [ ] All tests pass (including new test for this bug)
- [ ] Code review completed
- [ ] Documentation updated if needed
