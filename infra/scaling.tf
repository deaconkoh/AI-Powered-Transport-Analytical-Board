resource "aws_autoscaling_policy" "tt_req_per_target" {
  name                   = "${var.project_name}-tt-req-per-target"
  autoscaling_group_name = aws_autoscaling_group.api_asg.name
  policy_type            = "TargetTrackingScaling"

  target_tracking_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ALBRequestCountPerTarget"
      resource_label         = "${aws_lb.api_alb.arn_suffix}/${aws_lb_target_group.api_tg.arn_suffix}"
    }
    target_value = 100 # tune later
  }
}
