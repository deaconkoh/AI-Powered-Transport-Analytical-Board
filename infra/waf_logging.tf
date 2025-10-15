########################################
# WAF → CloudWatch Logs
########################################

# WAF requires the log group name to start with "aws-waf-logs-"
resource "aws_cloudwatch_log_group" "waf_logs" {
  name              = "aws-waf-logs-${var.project_name}" 
  retention_in_days = 14
}

# Attach WAF logging to your existing Web ACL
resource "aws_wafv2_web_acl_logging_configuration" "waf_to_cwlogs" {
  resource_arn          = aws_wafv2_web_acl.api_waf.arn
  log_destination_configs = [aws_cloudwatch_log_group.waf_logs.arn]

  redacted_fields {
    single_header { name = "authorization" }
  }
}