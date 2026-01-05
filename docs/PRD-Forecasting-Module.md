# GMP Line-Level Forecasting Module - Product Requirements Document

**Version:** 1.0
**Status:** Draft
**Last Updated:** 2026-01-04
**Author:** Product Team

---

## Summary

The Forecasting Module provides construction project managers with line-item-level cost forecasting capabilities for GMP (Guaranteed Maximum Price) contracts. Each GMP line item receives a dedicated forecast page displaying projected costs through completion in both weekly and monthly time buckets.

The module supports five forecasting methodologies—Earned Value Management (EVM), Three-Point Estimating (PERT), Parametric Estimating, Grey-Fuzzy Logic/Neural Networks, and Time Series (ARIMA)—selectable per GMP line based on data availability and user preference. Forecasts reconcile across time granularities (weekly totals sum to monthly) and roll up to project-level summaries while maintaining full auditability of inputs, methods, and calculation changes.

**Key Outcomes:**
- Early identification of cost overruns at the line-item level
- Improved cash flow planning through time-bucketed projections
- Configurable methodology selection based on data maturity and project phase
- Full traceability from forecast outputs back to source transactions and assumptions

---

## User Stories

### Primary Personas
- **Project Cost Controller:** Manages GMP reconciliation and forecasting for one or more projects
- **Project Manager:** Reviews forecasts for executive reporting and decision-making
- **Finance Analyst:** Uses forecasts for cash flow planning and variance analysis

### User Stories

| ID | As a... | I want to... | So that... | Priority |
|----|---------|--------------|------------|----------|
| US-01 | Project Cost Controller | View a dedicated forecast page for each GMP line item | I can analyze projected costs at the detail level | P0 |
| US-02 | Project Cost Controller | Toggle between weekly and monthly forecast views | I can match my reporting cadence without re-running calculations | P0 |
| US-03 | Project Cost Controller | Select the forecasting method for each GMP line | I can use the most appropriate method based on available data | P0 |
| US-04 | Project Manager | See EAC (Estimate at Completion) prominently displayed | I can quickly assess whether a line is trending over or under budget | P0 |
| US-05 | Finance Analyst | View remaining cost distributed across future periods | I can plan cash disbursements accurately | P0 |
| US-06 | Project Cost Controller | Understand how the forecast was calculated in plain language | I can validate assumptions and explain results to stakeholders | P1 |
| US-07 | Project Cost Controller | See forecasts automatically recalculate when transactions change | I always see current projections | P0 |
| US-08 | Project Manager | View project-level rollup of all line forecasts | I can see the total projected cost for executive reporting | P1 |
| US-09 | Finance Analyst | Export forecast data in CSV/Excel format | I can incorporate forecasts into external financial models | P2 |
| US-10 | Project Cost Controller | View forecast history and audit trail | I can track how projections have changed over time | P1 |

---

## In Scope / Out of Scope

### In Scope

| Item | Description |
|------|-------------|
| Line-level forecast pages | Dedicated forecast view for each GMP division/line item |
| Weekly time buckets | Monday-Sunday week boundaries with projection through completion |
| Monthly time buckets | Calendar month boundaries with projection through completion |
| Five forecasting methods | EVM, PERT, Parametric, Grey-Fuzzy/Neural, and ARIMA |
| Method selection per line | User can choose and change the forecasting method for each GMP line |
| Automatic recalculation | Forecasts update when source transactions or mappings change |
| Week-to-month reconciliation | Weekly buckets sum to monthly totals with cross-month allocation |
| Project-level rollups | Aggregated forecasts across all GMP lines |
| Audit trail | History of forecast changes with timestamps and triggers |
| Fallback handling | Graceful degradation when EV/SPI or other inputs are unavailable |
| Performance optimization | Usable performance for projects with >100,000 transactions |

### Out of Scope (v1.0)

| Item | Rationale |
|------|-----------|
| Monte Carlo simulation | Future enhancement; requires additional infrastructure |
| Resource-loaded scheduling integration | Requires integration with external scheduling tools (P6, MS Project) |
| Multi-project portfolio forecasting | Focus on single-project accuracy first |
| Forecast approval workflows | Can be added in v1.1 based on user feedback |
| Mobile-optimized views | Desktop-first; responsive but not mobile-native |
| Real-time collaboration | Single-user editing per line; no concurrent edit conflict resolution |
| Custom forecasting formulas | Only predefined methods; custom formulas in future versions |

---

## Functional Requirements

### 1. Forecast Page per GMP Line

**FR-1.1** Each GMP line item SHALL have a dedicated forecast page accessible from the GMP summary view.

**FR-1.2** The forecast page URL pattern SHALL be: `/gmp/{division}/forecast` where `{division}` is the URL-encoded GMP division identifier.

**FR-1.3** The forecast page header SHALL display:
- GMP Line identifier and description
- Budget at Completion (BAC)
- Actual Cost to Date (AC)
- Current EAC (Estimate at Completion)
- Variance (EAC - BAC) with color coding: green (under), yellow (±5%), red (over)
- Percent complete (by cost)
- Forecast completion date
- Active forecasting method with "Change" action

**FR-1.4** The forecast page SHALL display a time-bucketed table showing:
- Period identifier (Week # or Month name)
- Period date range
- Forecast remaining cost for period
- Cumulative forecast cost to date
- Variance vs. BAC at period end
- Confidence indicator (High/Medium/Low) where applicable

**FR-1.5** The page SHALL include a "Method Explanation" section displaying:
- Name of the active forecasting method
- Plain-language description of how the forecast was calculated
- Key inputs used (with values)
- Any assumptions or fallbacks applied
- Date/time of last calculation

### 2. Weekly Forecasting Rules

**FR-2.1** Weekly buckets SHALL use ISO 8601 week definitions: Monday 00:00:00 through Sunday 23:59:59.

**FR-2.2** The weekly forecast SHALL project from the current week through the forecast completion week.

**FR-2.3** Past weeks (completed) SHALL display actual costs, not forecasted values.

**FR-2.4** The current (partial) week SHALL display:
- Actual cost incurred so far (through latest transaction date)
- Forecasted remaining cost for the remainder of the week
- Blended total (actual + forecast)

**FR-2.5** Weeks that span two calendar months SHALL be handled as follows:
- The week's cost is allocated proportionally by day count to each month
- Example: A week spanning Jan 28 - Feb 3 has 4 days in January, 3 days in February
- 4/7 of that week's cost allocates to January, 3/7 to February

**FR-2.6** Weekly buckets SHALL be numbered sequentially from project start (Week 1, Week 2, etc.) with ISO week/year shown in tooltip.

### 3. Monthly Forecasting Rules

**FR-3.1** Monthly buckets SHALL align to calendar months (first day through last day).

**FR-3.2** The monthly forecast SHALL project from the current month through the forecast completion month.

**FR-3.3** Past months (completed) SHALL display actual costs, not forecasted values.

**FR-3.4** The current (partial) month SHALL display:
- Actual cost incurred so far
- Forecasted remaining cost for the remainder of the month
- Blended total (actual + forecast)

**FR-3.5** Monthly totals SHALL equal the sum of all weekly bucket values that fall within that month (after day-count allocation for spanning weeks per FR-2.5).

**FR-3.6** Rounding differences SHALL be allocated to the last week of the month to ensure perfect reconciliation.

### 4. Method Selection and Method-Specific Inputs

**FR-4.1** The system SHALL support the following five forecasting methods:

#### 4.1.1 Earned Value Management (EVM)

| Input | Source | Fallback |
|-------|--------|----------|
| Budget at Completion (BAC) | GMP line budget amount | Required; no fallback |
| Actual Cost (AC) | Sum of mapped direct cost transactions | Required; no fallback |
| Earned Value (EV) | External schedule integration OR user-entered percent complete × BAC | If unavailable: use AC as proxy with warning label |
| Schedule Performance Index (SPI) | External schedule integration OR user-entered | If unavailable: assume SPI = 1.0 with warning label |

**Calculations:**
- Cost Performance Index: `CPI = EV / AC` (if AC > 0; else undefined)
- Basic EAC: `EAC = BAC / CPI`
- Schedule-adjusted EAC: `EAC = AC + ((BAC - EV) / (CPI × SPI))`
- Remaining distribution: Linear across remaining periods unless burn rate pattern is available

**Output labeling:** If EV or SPI is derived/assumed rather than actual, display "(estimated)" suffix.

#### 4.1.2 Three-Point Estimating (PERT)

| Input | Source |
|-------|--------|
| Optimistic estimate (O) | User-entered per line |
| Most likely estimate (M) | User-entered per line |
| Pessimistic estimate (P) | User-entered per line |

**Calculations:**
- Expected value: `E = (O + 4M + P) / 6`
- Standard deviation: `σ = (P - O) / 6`
- EAC = E
- Confidence range: E ± σ (68%), E ± 2σ (95%)

**Remaining distribution:** User-selected curve (linear, front-loaded, back-loaded) across remaining periods.

#### 4.1.3 Parametric Estimating

| Input | Source |
|-------|--------|
| Quantity | User-entered or imported from quantity tracking |
| Unit rate | User-entered or calculated from historical actuals |
| Complexity factor | User-selected: Low (0.9), Normal (1.0), High (1.15), Very High (1.3) |
| Quantity adjustment factor (n) | Optional user override |

**Calculations:**
- Base estimate: `Base = Quantity × Unit Rate`
- Adjusted estimate: `EAC = Base × Complexity Factor × (1 + n)`
- Remaining = EAC - AC

**Remaining distribution:** Based on quantity installation schedule if available; else linear.

#### 4.1.4 Grey-Fuzzy Logic & Neural Networks

| Input | Source | Description |
|-------|--------|-------------|
| Historical cost patterns | Database | Cost curves from similar completed projects |
| Project characteristics | User-entered | Size, type, complexity classification |
| Environmental factors | User-entered | Location, market conditions, labor availability |
| Progress metrics | System | Percent complete, burn rate, timeline position |

**Model requirements:**
- Model SHALL be pre-trained on historical project data (minimum 20 comparable projects)
- Model weights SHALL NOT be generated or invented by this specification
- Model SHALL be trained/validated by data science team with documented accuracy metrics
- Minimum acceptable accuracy: 85% of predictions within ±10% of actual at 50% complete milestone

**Outputs:**
- Point estimate (EAC)
- Confidence interval (5th - 95th percentile)
- Feature importance ranking (which inputs drove the prediction)

**Fallback:** If model is not trained or confidence is below threshold, system SHALL disable this method for the line and notify user.

#### 4.1.5 Time Series (ARIMA)

| Input | Source | Description |
|-------|--------|-------------|
| Historical cost time series | Database | Minimum 12 periods of cost data for the line |
| Seasonality indicators | System-detected | Weekly/monthly patterns if present |
| Trend component | System-calculated | Direction and magnitude of cost trajectory |

**Model requirements:**
- ARIMA parameters (p, d, q) SHALL be determined by auto-selection algorithm (e.g., auto.arima)
- Model SHALL NOT use hardcoded or invented coefficients
- Minimum data requirement: 12 historical periods; fewer disables this method

**Outputs:**
- Point forecast per future period
- Confidence intervals (80%, 95%)
- Model diagnostics (AIC, residual analysis) available on request

**Fallback:** If insufficient historical data, system SHALL disable this method for the line with message: "Requires minimum 12 periods of historical data."

**FR-4.2** Method selection SHALL be persisted per GMP line and retained across sessions.

**FR-4.3** Changing the method SHALL trigger immediate recalculation and update the display.

**FR-4.4** The system SHALL recommend a method based on data availability with rationale.

### 5. Reconciliation and Rollups

**FR-5.1** Weekly-to-Monthly reconciliation:
- Sum of weekly forecast values allocated to a month SHALL equal the monthly forecast value for that month within $0.01 (rounding tolerance)
- Any rounding discrepancy SHALL be applied to the final week of the month

**FR-5.2** Line-to-Project rollup:
- Project-level EAC = Sum of all line-level EACs
- Project-level period forecast = Sum of all line-level period forecasts
- Rollup SHALL recalculate whenever any line forecast changes

**FR-5.3** Recalculation triggers:
| Trigger | Recalc Timing | Scope |
|---------|---------------|-------|
| New transaction posted | Near-real-time (< 5 minutes) | Affected GMP line |
| Transaction modified | Near-real-time (< 5 minutes) | Affected GMP line |
| Mapping changed | Near-real-time (< 5 minutes) | Affected GMP line(s) |
| Method changed | Immediate | Single GMP line |
| Completion date changed | Immediate | Single GMP line |
| Manual refresh requested | Immediate | As requested (line or project) |
| Scheduled batch | Daily at 02:00 local | All lines (full reconciliation) |

**FR-5.4** Audit trail requirements:
- Every forecast recalculation SHALL log: timestamp, trigger, previous EAC, new EAC, method, user (if manual)
- Audit log SHALL be retained for 7 years minimum
- User SHALL be able to view forecast history graph showing EAC over time

---

## UX and Accessibility Rules

**UX-1** Time granularity toggle (Weekly/Monthly) SHALL be a prominent toggle control, not buried in settings.

**UX-2** Toggling granularity SHALL NOT cause page reload; SHALL use client-side view switching with smooth transition.

**UX-3** Current period (week or month) SHALL be visually highlighted to distinguish from past and future periods.

**UX-4** Variance SHALL use consistent color coding throughout:
- Green: Favorable (under budget by >1%)
- Yellow: Neutral (within ±1%)
- Red: Unfavorable (over budget by >1%)

**UX-5** Method selection SHALL use a modal or slide-out panel showing:
- All five methods with availability status (enabled/disabled with reason)
- Recommended method highlighted
- Preview of EAC with selected method before confirming

**UX-6** All forecast tables SHALL support:
- Keyboard navigation (arrow keys, Tab)
- Screen reader labels for all data cells
- High contrast mode compatibility
- Minimum touch target size of 44×44px for interactive elements

**UX-7** Loading states:
- Display skeleton loader during initial load
- Show inline spinner during recalculation without blocking the full page
- Display "Last updated: [timestamp]" footer on forecast tables

**UX-8** Export function SHALL support:
- CSV format with headers matching display columns
- Excel format with formatting preserved
- Date range selection for export (all periods, past only, future only)

**UX-9** Responsive behavior:
- Tables SHALL horizontally scroll on viewports < 1024px
- Critical metrics (EAC, BAC, Variance) SHALL remain visible without scrolling
- Toggle and method selector SHALL stack vertically on mobile

---

## Data, Calculations, and API Requirements

### Data Model Extensions

```
Table: forecast_config
- id: UUID (PK)
- gmp_division: VARCHAR(100) (FK)
- method: ENUM('EVM', 'PERT', 'PARAMETRIC', 'NEURAL', 'ARIMA')
- method_params: JSONB (method-specific parameters)
- completion_date: DATE
- created_at: TIMESTAMP
- updated_at: TIMESTAMP
- created_by: VARCHAR(100)

Table: forecast_snapshot
- id: UUID (PK)
- gmp_division: VARCHAR(100) (FK)
- snapshot_date: DATE
- eac: DECIMAL(15,2)
- bac: DECIMAL(15,2)
- ac: DECIMAL(15,2)
- ev: DECIMAL(15,2) NULLABLE
- method: VARCHAR(20)
- confidence_level: VARCHAR(10)
- created_at: TIMESTAMP

Table: forecast_period
- id: UUID (PK)
- forecast_snapshot_id: UUID (FK)
- period_type: ENUM('WEEK', 'MONTH')
- period_start: DATE
- period_end: DATE
- forecast_amount: DECIMAL(15,2)
- actual_amount: DECIMAL(15,2) NULLABLE
- cumulative_forecast: DECIMAL(15,2)
- variance_from_bac: DECIMAL(15,2)
- created_at: TIMESTAMP

Table: forecast_audit_log
- id: UUID (PK)
- gmp_division: VARCHAR(100)
- event_type: VARCHAR(50)
- trigger: VARCHAR(100)
- previous_eac: DECIMAL(15,2)
- new_eac: DECIMAL(15,2)
- method: VARCHAR(20)
- user_id: VARCHAR(100) NULLABLE
- transaction_ids: JSONB NULLABLE
- created_at: TIMESTAMP
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/gmp/{division}/forecast` | Get current forecast for a GMP line |
| GET | `/api/gmp/{division}/forecast/periods?granularity={weekly\|monthly}` | Get time-bucketed forecast |
| PUT | `/api/gmp/{division}/forecast/method` | Update forecasting method |
| PUT | `/api/gmp/{division}/forecast/params` | Update method-specific parameters |
| POST | `/api/gmp/{division}/forecast/refresh` | Force recalculation |
| GET | `/api/gmp/{division}/forecast/history` | Get forecast change history |
| GET | `/api/gmp/{division}/forecast/export?format={csv\|xlsx}` | Export forecast data |
| GET | `/api/project/forecast/rollup` | Get project-level aggregated forecast |

### Request/Response Examples

**GET /api/gmp/Concrete/forecast**
```json
{
  "gmp_division": "Concrete",
  "as_of_date": "2026-01-04",
  "bac": 2500000.00,
  "ac": 1200000.00,
  "ev": 1150000.00,
  "eac": 2608695.65,
  "variance": 108695.65,
  "variance_percent": 4.35,
  "percent_complete": 46.0,
  "method": "EVM",
  "method_label": "Earned Value Management",
  "completion_date": "2026-08-15",
  "confidence": "MEDIUM",
  "last_calculated": "2026-01-04T10:30:00Z",
  "inputs": {
    "cpi": 0.958,
    "spi": 1.02,
    "spi_source": "estimated"
  },
  "explanation": "Based on Earned Value Management. Current CPI of 0.958 indicates spending 4.2% more than planned for work completed. SPI is estimated at 1.02 (user-entered). EAC calculated as AC + (BAC - EV) / (CPI × SPI)."
}
```

### Performance Requirements

| Metric | Target | Measurement |
|--------|--------|-------------|
| Forecast page initial load | < 2 seconds | P95 latency |
| Granularity toggle response | < 500ms | P95 latency |
| Single-line recalculation | < 3 seconds | P95 latency |
| Full project recalculation | < 30 seconds | P95 for projects with 50 GMP lines |
| Transaction volume support | 500,000 transactions | No degradation in above metrics |
| Concurrent users | 50 users | No degradation in above metrics |

### Calculation Precision

- All monetary calculations SHALL use DECIMAL(15,2) precision
- Intermediate calculations SHALL maintain at least 6 decimal places
- Final displayed values SHALL be rounded to 2 decimal places using banker's rounding
- Percentages SHALL display to 1 decimal place (e.g., 4.3%)

---

## Acceptance Criteria (Given/When/Then)

### AC-01: Weekly/Monthly Toggle Reconciliation
```gherkin
Given a GMP line "Concrete" with forecasted EAC of $2,500,000
And 16 weeks remaining until completion spanning 4 months
When the user views the Monthly forecast
And notes the total forecast remaining is $1,300,000
And the user toggles to Weekly view
Then the sum of all weekly forecast remaining values SHALL equal $1,300,000
And toggling back to Monthly SHALL show the same $1,300,000 total
```

### AC-02: Week Spanning Two Months - Day Count Allocation
```gherkin
Given a week (Jan 27 - Feb 2) with forecasted cost of $70,000
And 5 days fall in January (Jan 27-31) and 2 days fall in February (Feb 1-2)
When the user views the Monthly forecast for January
Then January's total SHALL include $50,000 from this week (5/7 × $70,000)
When the user views the Monthly forecast for February
Then February's total SHALL include $20,000 from this week (2/7 × $70,000)
And the sum ($50,000 + $20,000) SHALL equal the weekly amount ($70,000)
```

### AC-03: Missing EV Fallback to AC Proxy
```gherkin
Given a GMP line "Electrical" with BAC of $800,000 and AC of $400,000
And Earned Value (EV) is not available from any source
When the user selects EVM as the forecasting method
Then the system SHALL use AC ($400,000) as a proxy for EV
And the forecast page SHALL display "(EV estimated from AC)" indicator
And the confidence level SHALL be set to "LOW"
And the explanation SHALL state "Earned Value not available; using Actual Cost as proxy which assumes work completed equals money spent."
```

### AC-04: Missing SPI Fallback
```gherkin
Given a GMP line with EVM method selected
And Schedule Performance Index (SPI) is not available
When the forecast is calculated
Then the system SHALL assume SPI = 1.0
And display "(SPI assumed = 1.0)" indicator
And use the basic EAC formula (BAC / CPI) instead of schedule-adjusted formula
```

### AC-05: Recalculation After New Transaction
```gherkin
Given a GMP line "Masonry" with current AC of $500,000 and EAC of $1,100,000
When a new direct cost transaction of $25,000 is posted and mapped to "Masonry"
Then within 5 minutes the forecast SHALL automatically recalculate
And the AC SHALL update to $525,000
And the EAC SHALL recalculate based on the new AC
And an audit log entry SHALL be created with trigger = "transaction_posted"
```

### AC-06: Recalculation After Mapping Change
```gherkin
Given a direct cost transaction of $15,000 currently mapped to GMP line "Plumbing"
And "Plumbing" has EAC of $600,000
And "Electrical" has EAC of $800,000
When the user changes the mapping from "Plumbing" to "Electrical"
Then both "Plumbing" and "Electrical" forecasts SHALL recalculate within 5 minutes
And "Plumbing" AC SHALL decrease by $15,000
And "Electrical" AC SHALL increase by $15,000
And audit log entries SHALL be created for both lines
```

### AC-07: Completion Date Change - Period Redistribution
```gherkin
Given a GMP line "HVAC" with completion date of June 30, 2026
And remaining forecast of $400,000 distributed across 6 months (Jan-Jun)
When the user changes the completion date to August 31, 2026
Then the remaining $400,000 SHALL be redistributed across 8 months (Jan-Aug)
And each monthly bucket SHALL be recalculated proportionally
And the EAC total SHALL remain $400,000 (unchanged)
```

### AC-08: PERT Method Calculation
```gherkin
Given a GMP line "Fire Protection" using PERT method
And user enters Optimistic = $200,000, Most Likely = $250,000, Pessimistic = $350,000
When the forecast is calculated
Then EAC SHALL equal ($200,000 + 4×$250,000 + $350,000) / 6 = $258,333.33
And standard deviation SHALL equal ($350,000 - $200,000) / 6 = $25,000
And 95% confidence range SHALL display as $208,333 to $308,333
```

### AC-09: Parametric Method with Complexity Factor
```gherkin
Given a GMP line "Drywall" using Parametric method
And Quantity = 50,000 SF, Unit Rate = $8.50/SF, Complexity = High (1.15)
When the forecast is calculated
Then Base estimate = 50,000 × $8.50 = $425,000
And EAC = $425,000 × 1.15 = $488,750
```

### AC-10: ARIMA Method Insufficient Data
```gherkin
Given a GMP line "Roofing" with only 8 periods of historical cost data
When the user attempts to select ARIMA as the forecasting method
Then the ARIMA option SHALL be displayed as disabled
And a tooltip SHALL explain "Requires minimum 12 periods of historical data (currently 8)"
And the user SHALL NOT be able to select ARIMA until sufficient data exists
```

### AC-11: Project-Level Rollup Accuracy
```gherkin
Given 5 GMP lines with the following EACs:
  - Concrete: $2,500,000
  - Masonry: $1,100,000
  - Electrical: $800,000
  - Plumbing: $600,000
  - HVAC: $750,000
When the user views the project-level forecast rollup
Then the project EAC SHALL equal $5,750,000
And each period's project forecast SHALL equal the sum of line forecasts for that period
```

### AC-12: Audit Trail Completeness
```gherkin
Given a GMP line "Concrete" that has undergone 5 forecast changes over 30 days
When the user views the forecast history
Then all 5 changes SHALL be displayed in reverse chronological order
And each entry SHALL show: date, trigger, previous EAC, new EAC, method, user (if manual)
And the user SHALL be able to view a line chart of EAC over time
```

### AC-13: Zero AC Edge Case
```gherkin
Given a GMP line "Site Work" with BAC of $300,000 and AC of $0
And work has not yet started
When EVM method is selected
Then CPI SHALL display as "N/A" (division by zero avoided)
And EAC SHALL default to BAC ($300,000)
And explanation SHALL state "No costs incurred yet; forecast equals budget."
```

### AC-14: Negative Adjustment Handling
```gherkin
Given a GMP line "Concrete" with AC of $500,000
When a credit/adjustment transaction of -$20,000 is posted
Then AC SHALL update to $480,000
And the forecast SHALL recalculate with the reduced AC
And the audit log SHALL capture the negative transaction appropriately
```

---

## Edge Cases and Test Notes

### Edge Case Matrix

| ID | Scenario | Expected Behavior |
|----|----------|-------------------|
| EC-01 | Zero AC (no costs yet) | CPI undefined; EAC = BAC; clear messaging |
| EC-02 | AC > BAC (already over budget) | CPI < 1; EAC > BAC; red variance display |
| EC-03 | EV > AC (ahead of schedule/under cost) | CPI > 1; EAC < BAC; green variance |
| EC-04 | Negative AC (credits exceed costs) | Handle gracefully; may indicate data issue |
| EC-05 | Completion date in past | Flag as "Past Due"; show actuals only |
| EC-06 | Completion date today | Current period is final period |
| EC-07 | Inactive GMP line | Exclude from rollups; show "Inactive" badge |
| EC-08 | Late-posted transaction (dated in past month) | Recalculate affected historical period; flag in audit |
| EC-09 | Partial week at project start | Pro-rate first week's forecast |
| EC-10 | Partial week at project end | Pro-rate final week's forecast |
| EC-11 | Very large transaction volume (>100K) | Pagination/lazy load; background recalculation |
| EC-12 | Concurrent edits to same line | Last-write-wins with conflict notification |
| EC-13 | Method change mid-period | Apply from current period forward; don't restate history |
| EC-14 | All methods unavailable | Display "Insufficient data for forecasting" with guidance |
| EC-15 | Rounding accumulation error | Allocate residual to final period |

### Performance Test Scenarios

| Test | Parameters | Pass Criteria |
|------|------------|---------------|
| Load test - page render | 50 concurrent users, 100 GMP lines | P95 < 2s |
| Stress test - recalculation | 500K transactions, single line recalc | P95 < 3s |
| Endurance test | 8-hour continuous usage, 30 users | No memory leak, consistent response times |
| Spike test | 0 to 100 users in 1 minute | Graceful degradation, no errors |

### Data Quality Validations

- Transaction amounts must be numeric (reject non-numeric)
- Dates must be valid and within project date range
- GMP line mappings must reference valid divisions
- PERT estimates must satisfy O ≤ M ≤ P
- Unit rates must be positive numbers
- Complexity factors must be within valid range (0.5 - 2.0)

---

## Open Questions

| ID | Question | Owner | Status |
|----|----------|-------|--------|
| OQ-01 | Should forecast history be editable/deletable by admins? | Product | Open |
| OQ-02 | What is the source system for EV and SPI integration? P6? MS Project? Manual entry only? | Engineering | Open |
| OQ-03 | For Neural Network method, who provides the training data and what is the model update cadence? | Data Science | Open |
| OQ-04 | Should week boundaries be configurable (e.g., Sun-Sat for some clients)? | Product | Open |
| OQ-05 | What is the retention policy for forecast_period table given high data volume? | Engineering | Open |
| OQ-06 | Should ARIMA auto-select parameters on every recalculation or lock after initial fit? | Data Science | Open |
| OQ-07 | How should forecasts handle change orders that modify BAC? | Product | Open |
| OQ-08 | Is there a need for forecast "what-if" scenarios without saving? | Product | Open |
| OQ-09 | Should users be able to manually override calculated EAC? If so, how is this tracked? | Product | Open |
| OQ-10 | What notification/alert thresholds should trigger when forecast exceeds budget by X%? | Product | Open |

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **BAC** | Budget at Completion - the approved budget for a GMP line |
| **AC** | Actual Cost - costs incurred to date from mapped transactions |
| **EV** | Earned Value - value of work actually completed |
| **EAC** | Estimate at Completion - forecasted total cost at project end |
| **CPI** | Cost Performance Index - efficiency metric (EV/AC) |
| **SPI** | Schedule Performance Index - schedule efficiency metric |
| **GMP** | Guaranteed Maximum Price - contract type with cost ceiling |
| **PERT** | Program Evaluation and Review Technique |
| **ARIMA** | AutoRegressive Integrated Moving Average |

---

## Appendix B: Method Selection Decision Tree

```
Start
  │
  ├─ Is historical time-series data available (≥12 periods)?
  │    ├─ Yes → Consider ARIMA
  │    └─ No → Skip ARIMA
  │
  ├─ Is EV available from scheduling system?
  │    ├─ Yes → EVM Recommended
  │    └─ No → Can user estimate % complete?
  │              ├─ Yes → EVM with estimated EV
  │              └─ No → Skip EVM or use AC proxy
  │
  ├─ Does user have O/M/P estimates?
  │    ├─ Yes → Consider PERT
  │    └─ No → Skip PERT
  │
  ├─ Is quantity tracking available?
  │    ├─ Yes → Consider Parametric
  │    └─ No → Skip Parametric
  │
  ├─ Is trained ML model available for this project type?
  │    ├─ Yes → Consider Neural Network
  │    └─ No → Skip Neural Network
  │
  └─ Default: EVM with fallbacks or PERT with user estimates
```

---

*End of Document*
