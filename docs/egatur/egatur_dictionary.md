### 🔹 Variables in `elevado_eg_mod_web_tur_*.txt` (trips)

| Variable       | Type     | Values / Example   | Description |
|----------------|----------|-------------------|-------------|
| `mm_aaaa`      | string   | `0123`            | Reference month and year (MMYY). |
| `A0`           | string   | `2`               | Survey source (2 = Egatur). |
| `A0_1`         | string   | `20200200009238`  | **Unique identifier** of the questionnaire (primary key for joins). |
| `A0_7`         | string   | `2`, `8`          | Tourist type: <br>• 2 = Non-resident (non-transit) <br>• 8 = Non-resident in transit. |
| `A1`           | string   | `1-4`             | Exit route: <br>1 = Road, 2 = Airport, 3 = Port, 4 = Train. |
| `pais`         | string   | `01-15`           | Country of residence: <br>01 = Germany, 02 = Belgium, …, 15 = Rest of the world. |
| `ccaa`         | string   | `01-19`           | Main destination Autonomous Community (region): <br>01 = Andalusia, …, 19 = Melilla. |
| `A13`          | integer  | `3`               | Total overnight stays in the trip. |
| `aloja`        | string   | `1-3`             | Main accommodation type: <br>1 = Hotels and similar, 2 = Other market, 3 = Non-market. |
| `motivo`       | string   | `1-3`             | Main trip purpose: <br>1 = Leisure/holidays, 2 = Business, 3 = Other. |
| `A16`          | string   | `1`, `6`          | Package tour: 1 = Yes, 6 = No. |
| `gastototal`   | decimal  | `2341.84`         | Total expenditure of the trip/excursion. |
| `factoregatur` | decimal  | `1998.09`         | Expansion factor (sampling weight). |

**Methodological notes:**
- **Estimated tourist expenditure** = `gastototal * factoregatur`.
- **Number of tourists** = sum of `factoregatur`.
- **Estimated overnight stays** = `A13 * factoregatur`.
- **Daily average expenditure** = total expenditure / estimated overnight stays.
- **Average trip length** = estimated overnight stays / tourists.