# Installation Request Form Trigger with Automated Slack Message using n8n

### What is this project?
* A hands-on exercise demonstrating flow-based programming fundamentals using n8n (a low-code platform for building automations), to process installation requests through conditional routing and send a Slack notification to a team channel, marked for urgent attention. 

### Tools Used:
* `n8n`
* `JavaScript`
* `Slack API` 
* `Oauth2`
* `JSON`
* `HTTP Webhook`

### Usage
1. `Form Access`: Use the generated public URL to submit installation requests (e.g. provided by a B2B SaaS)
2. `Urgent Requests`: Dates within 7 days trigger immediate Slack notification
3. `Non-Urgent`: Requests outside the window require no immediate action
4. `Monitoring`: Check execution logs for workflow performance

### [Link to Installation Form Example](https://sevenlabtech.app.n8n.cloud/form/1874a12d-2889-419d-ac29-52151ca3bf82)

## Overview

This project implements a two-branch workflow system that:
- Starts with a **web form submission trigger**
- Uses `conditional logic` to route urgent vs. non-urgent requests  
- Automatically notifies a Slack channel, such as the `sales team`, for urgent requests
- Demonstrates core automation principles, including triggers, actions, data mapping, and environment management

## Problem Statement

Manual processes for handling incoming requests are both time-consuming and prone to errors. Specifically, this workflow addresses:

- **Manual Request Handling**: Users submit installation requests via a web form, requiring manual team notification
- **Priority Filtering**: Need to identify and prioritise urgent requests (installation date within 7 days)
- **Communication Overhead**: Manual routing of information to appropriate team members

## Solution Architecture

### Workflow Components

1. **Form Trigger** - Public web form entry point
2. **Conditional Logic** - Date-based routing for urgency
3. **Slack Integration** - Automated team notifications
4. **Data Flow** - JSON payload processing between nodes

### Technical Implementation

#### Step 1: Form Trigger Setup
- **Node Type**: `On Form Submission` trigger
- **Configuration**: 
  - Title: "Request an Installation"
  - Fields: 
    - `Email` (validated)
    - `Preferred Install Date` (date type)
- **Output**: Generates a unique shareable URL and captures form data as **JSON**:
```json 
[
  {
    "Email Address": "name@gmail.com",
    "Preferred Install Date": "2025-08-27",
    "submittedAt": "2025-08-23T15:05:36.458+01:00",
    "formMode": "production"
  }
]
```

<img width="1337" height="375" alt="Image" src="https://github.com/user-attachments/assets/7e609a16-3dc0-4117-a480-41a66561aa80" />

<img width="2773" height="1808" alt="Image" src="https://github.com/user-attachments/assets/2a8b2ac3-14c7-4517-8d30-295a457f7bef" />

#### Step 2: Data Management
- **Pinning Strategy**: Save test payload to node for development efficiency
- **Data Format**: JSON structure with email and date fields
- **Testing**: Reusable test data without form resubmission

#### Step 3: Conditional Logic Implementation

```javascript
// Conditional statement setup reference time (date of instalment)
{{ $json['Preferred Install Date'].toDateTime() }} 
```

```javascript
// Date comparison expression (if install date is before or equal to now + 7 days)
$now.plus(7, 'days').toDatetime()
```
- **Logic**: If `preferredInstallDate` ≤ (current date + 7 days)
- **Routing**: True branch → Slack notification, False branch → No action

<img width="1353" height="371" alt="Image" src="https://github.com/user-attachments/assets/3b415784-77f8-4530-8704-866f6ba5be4d" />

<img width="3296" height="761" alt="Image" src="https://github.com/user-attachments/assets/6ba183eb-e200-4a2a-91cd-be22be668614" />

#### Step 4: Slack Integration
- **Configuration**:
  - Resource: Channel
  - Operation: Send Message
  - Destination: #sales channel (example)
    - In the demo, I routed it to `#7lab-task-tracker`: a sandbox channel 
- **Message Composition**: Dynamic content using data mapping
- **Data Mapping**: Form fields → Message template

<img width="1005" height="302" alt="Image" src="https://github.com/user-attachments/assets/4a02c988-b341-4340-afd7-f73f60476c24" />

<img width="2754" height="1426" alt="Image" src="https://github.com/user-attachments/assets/bd6788e1-04e2-4dbd-afa7-1cd16c0bbc93" />

<img width="2252" height="428" alt="Image" src="https://github.com/user-attachments/assets/67a7cc83-2f25-4e12-87c3-cfdbc003d4cf" />

#### Step 5: Deployment
- **Activation**: Workflow goes live for production data
- **Monitoring**: Execution logs track production runs
- **Fallback**: No-op node for non-urgent requests

<img width="3044" height="694" alt="Image" src="https://github.com/user-attachments/assets/a11fe33d-589d-4080-8b01-ba50f4731213" />

<img width="1566" height="136" alt="Image" src="https://github.com/user-attachments/assets/13b19518-83e5-4b15-9336-d79cec230286" />

## Key Learnings

### Flow-Based Programming Concepts
- **Workflow Structure**: Single trigger → Multiple action nodes
- **Data Flow**: Items passed between nodes as arrays
- **Data Mapping**: Dynamic field insertion using expressions in `JavaScript`
- **Testing**: Pin data mechanism for development efficiency

### Technical Skills Developed
- **Conditional Logic**: Date calculations and boolean routing
- **API Integration**: Slack webhook configuration
- **Data Transformation**: JSON payload manipulation
- **Environment Management**: Test vs. production execution

## Future Applications

This foundation enables more sophisticated automations for AI engineering contexts:
- **Event-Driven Processing**: Trigger workflows from various data sources
- **Conditional Orchestration**: Complex routing based on multiple criteria  
- **Integration Hub**: Connect diverse tools and services
- **Data Pipeline**: Transform and route information between systems

## Technical Stack

- **Platform**: n8n (Low-code automation)
- **Integrations**: Slack API
- **Data Format**: JSON
- **Trigger Type**: HTTP webhook (form submission)
- **Logic**: JavaScript expressions for date manipulation

## Project Status

✅ **Completed**: Functional workflow deployed and tested  
✅ **Validated**: End-to-end automation confirmed  
✅ **Documented**: Implementation steps and learnings captured

---
