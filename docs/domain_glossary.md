# Domain Glossary

## Admission Status

In the Phase 1 mock data, `dateEnd: null` means the patient is currently
admitted. If `dateEnd` contains a timestamp, that admission is closed and the
patient is discharged.

## Diagnosis Codes

The mock data uses Italian ministerial ICD-9-CM numeric diagnosis codes. These
are not ICD-10 codes. Prefix matching such as `428` is used only for the Phase 1
cohort demo.
