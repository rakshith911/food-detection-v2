/**
 * AWS Cognito Custom Message Lambda
 *
 * Attach this to your User Pool:
 * Cognito → Your user pool → Extensions → Lambda triggers → Custom message → select this function.
 *
 * Trigger sources handled:
 *   CustomMessage_SignUp / CustomMessage_ResendCode  → sign-in OTP email (new users)
 *   CustomMessage_ForgotPassword                     → generic verification code email
 *                                                       (covers both existing-user sign-in
 *                                                        AND delete-account, since both call
 *                                                        resetPassword in the app)
 *
 * Logo: upload PHOTO-2026-02-22-22-02-16.jpg to S3 as logo-full.png with public read.
 * URL: https://ukcal-assets.s3.amazonaws.com/logo-full.png
 */

const LOGO_URL = 'https://ukcal-assets.s3.amazonaws.com/logo-full.png';

// ── Email template ────────────────────────────────────────────────────────────

function buildEmail() {
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="color-scheme" content="light">
<meta name="supported-color-schemes" content="light">
</head>
<body style="margin: 0; padding: 0; font-family: 'Segoe UI', Arial, sans-serif; background-color: #f9fafc;">
<table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%" style="background-color: #f9fafc;">
<tr><td style="padding: 40px 20px;">
<table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%" style="max-width: 460px; margin: 0 auto; background-color: #ffffff; border-radius: 20px; border: 1.5px solid #7ba21b; overflow: hidden;">

  <!-- Logo Section -->
  <tr>
    <td align="center" bgcolor="#eef7e2" style="padding: 44px 40px 40px 40px; text-align: center; background-color: #eef7e2; border-radius: 18px 18px 0 0;">
      <img src="${LOGO_URL}" alt="UKcal" width="240" style="display: block; width: 240px; height: auto; margin: 0 auto; background-color: #eef7e2;" />
    </td>
  </tr>

  <!-- Divider -->
  <tr>
    <td bgcolor="#7ba21b" style="background-color: #7ba21b; padding: 0; height: 1px; line-height: 1px; font-size: 1px;">&#8203;</td>
  </tr>

  <!-- Code Section -->
  <tr>
    <td bgcolor="#ffffff" style="padding: 40px 32px 28px 32px; background-color: #ffffff;">
      <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%" style="background-color: #eef7e2; border: 1.5px solid #7ba21b; border-radius: 14px;">
        <tr>
          <td style="padding: 24px 20px; text-align: center;">
            <p style="margin: 0 0 14px 0; color: #5a7a10; font-size: 18px; font-weight: 700; font-family: 'Segoe UI', Arial, sans-serif;">Your Code</p>
            <p style="margin: 0; color: #5a7a10; font-size: 38px; font-weight: 700; letter-spacing: 12px; font-family: 'Courier New', monospace;">{####}</p>
          </td>
        </tr>
      </table>
    </td>
  </tr>

  <!-- Expiry + Disclaimer -->
  <tr>
    <td bgcolor="#ffffff" style="padding: 0 32px 28px 32px; background-color: #ffffff; text-align: center;">
      <p style="margin: 0; color: #1f2937; font-size: 13px; font-weight: 600; line-height: 1.8; text-align: center;">
        This code will expire in 5 minutes.<br />
        If you did not request this code, you can safely ignore this email.
      </p>
    </td>
  </tr>

  <!-- Footer -->
  <tr>
    <td bgcolor="#ffffff" style="padding: 16px 32px 28px 32px; background-color: #ffffff;">
      <p style="margin: 0 0 6px 0; color: #6b7280; font-size: 12px; text-align: center;">© 2025 UKcal. All rights reserved.</p>
      <p style="margin: 0; color: #9ca3af; font-size: 11px; text-align: center;">This is an automated message. Please do not reply.</p>
    </td>
  </tr>

</table>
</td></tr>
</table>
</body>
</html>`;
}

// ── Handler ──────────────────────────────────────────────────────────────────

exports.handler = async (event) => {
  const trigger = event.triggerSource;

  if (
    trigger === 'CustomMessage_SignUp' ||
    trigger === 'CustomMessage_ResendCode' ||
    trigger === 'CustomMessage_ForgotPassword'
  ) {
    event.response.emailSubject = 'UKCal One-Time-Password';
    event.response.emailMessage = buildEmail();
    return event;
  }

  // All other triggers — use Cognito defaults
  return event;
};
