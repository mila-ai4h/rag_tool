# Required Permissions for Deploying the Infrastructure

Below is a table summarizing the required roles and permissions needed to deploy the infrastructure:

| **Role Name**                  | **Role ID**                                | **Description**                                                                              |
|--------------------------------|--------------------------------------------|----------------------------------------------------------------------------------------------|
| Editor                         | `roles/editor`                             | Provides broad permissions for resource management and operations within the project.         |
| Project IAM Admin              | `roles/resourcemanager.projectIamAdmin`    | Allows management of IAM policies, including assigning and revoking roles for project members.|
| Secret Manager Secret Accessor | `secretmanager.versions.access`            | Grants access to secret versions stored in Secret Manager to retrieve sensitive data.           |
| Cloud Run Admin                | `roles/run.admin`                          | Permits creation, update, and management of Cloud Run services.                                |

Ensure that these roles are properly assigned before initiating the deployment process.