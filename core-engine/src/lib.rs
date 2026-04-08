use pyo3::prelude::*;
use serde::Deserialize;
use std::process::Command;

#[pyfunction]
fn get_version() -> PyResult<String> {
    Ok("0.2.0".to_string())
}

#[pyfunction]
fn extract_metadata(path: String) -> PyResult<String> {
    let output = Command::new("ffprobe")
        .args(&[
            "-v", "error",
            "-show_entries", "format=duration:stream=width,height,avg_frame_rate",
            "-of", "json",
            &path,
        ])
        .output()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to run ffprobe: {}", e)))?;

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// A keyframe for dynamic panning: relative time (seconds from clip start) + center_x (0.0-1.0)
#[derive(Deserialize, Debug, Clone)]
struct PanKeyframe {
    t: f32,  // relative time from clip start (seconds)
    x: f32,  // normalized center_x (0.0 - 1.0)
}

/// Build a static crop filter expression with boundary clamping (RS-2 fix)
fn build_static_crop(center_x_norm: f32) -> String {
    // Clamp: x = max(0, min(iw - crop_w, iw * cx - crop_w/2))
    format!(
        "crop=ih*9/16:ih:max(0\\,min(iw-ih*9/16\\,iw*{:.4}-ih*9/16/2)):0,scale=1080:1920",
        center_x_norm.clamp(0.05, 0.95)
    )
}

/// Build a dynamic crop filter expression with piecewise-linear interpolation.
///
/// Includes RS-1 fix: limits keyframes to 20 to prevent FFmpeg expression depth overflow.
/// Includes RS-2 fix: wraps in clip() for boundary clamping.
fn build_dynamic_crop(keyframes: &[PanKeyframe]) -> String {
    if keyframes.is_empty() {
        return build_static_crop(0.5);
    }
    if keyframes.len() == 1 {
        return build_static_crop(keyframes[0].x);
    }

    // RS-1 fix: Limit keyframes to prevent expression depth overflow
    let max_kf: usize = 20;
    let kfs: Vec<PanKeyframe> = if keyframes.len() > max_kf {
        let mut sampled = vec![keyframes[0].clone()];
        let step = (keyframes.len() - 1) as f64 / (max_kf - 1) as f64;
        for i in 1..max_kf - 1 {
            sampled.push(keyframes[(i as f64 * step) as usize].clone());
        }
        sampled.push(keyframes[keyframes.len() - 1].clone());
        sampled
    } else {
        keyframes.to_vec()
    };

    let x_expr = build_lerp_expression(&kfs);

    // RS-2 fix: Wrap in clip(0, iw-crop_w, expr) for boundary safety
    format!(
        "crop=ih*9/16:ih:clip(0\\,iw-ih*9/16\\,{}):0",
        x_expr
    )
}

/// Build a nested FFmpeg expression for piecewise-linear interpolation.
///
/// Generates: if(lt(t,T1), X0, if(lt(t,T2), lerp(X1,X2,t), ... , Xn))
///
/// The expression converts normalized x (0.0-1.0) to pixel position:
///   pixel_x = iw * x_norm - crop_width / 2
///   where crop_width = ih * 9 / 16
fn build_lerp_expression(keyframes: &[PanKeyframe]) -> String {
    let n = keyframes.len();
    if n == 0 {
        return "crop=ih*9/16:ih:iw*0.5-ih*9/16/2:0,scale=1080:1920".to_string();
    }

    // Base case: the final x value
    let last = &keyframes[n - 1];
    let mut x_expr = format!("iw*{:.4}-ih*9/16/2", last.x);

    // Build the nested if(lt(t,T), ...) expression for the 'x' parameter
    for i in (0..n - 1).rev() {
        let kf_start = &keyframes[i];
        let kf_end = &keyframes[i + 1];

        let t_start = kf_start.t;
        let t_end = kf_end.t;
        let x_start = kf_start.x;
        let x_end = kf_end.x;

        let dt = t_end - t_start;

        if dt < 0.01 || (x_end - x_start).abs() < 0.001 {
            // Constant position before t_end
            x_expr = format!(
                "if(lt(t\\,{:.3})\\,iw*{:.4}-ih*9/16/2\\,{})",
                t_end, x_start, x_expr
            );
        } else {
            // Linear interpolation between t_start and t_end
            let lerp_val = format!(
                "iw*({:.4}+{:.4}*(t-{:.3})/{:.3})-ih*9/16/2",
                x_start,
                x_end - x_start,
                t_start,
                dt
            );
            x_expr = format!(
                "if(lt(t\\,{:.3})\\,iw*{:.4}-ih*9/16/2\\,if(lt(t\\,{:.3})\\,{}\\,{}))",
                t_start,
                x_start,
                t_end,
                lerp_val,
                x_expr
            );
        }
    }

    format!("crop=ih*9/16:ih:{}:0,scale=1080:1920", x_expr)
}

#[pyfunction]
#[pyo3(signature = (input_path, output_path, start, duration, center_x_norm, subtitle_path=None, keyframes_json=None, watermark_json_str=None))]
fn render_clip(
    input_path: String,
    output_path: String,
    start: f32,
    duration: f32,
    center_x_norm: f32,           // Static fallback (0.0 to 1.0)
    subtitle_path: Option<String>,
    keyframes_json: Option<String>, // Dynamic panning: JSON array of {"t": float, "x": float}
    watermark_json_str: Option<String>,
) -> PyResult<String> {
    // Determine crop filter: dynamic (if keyframes provided) or static
    let crop_filter = if let Some(ref kf_json) = keyframes_json {
        match serde_json::from_str::<Vec<PanKeyframe>>(kf_json) {
            Ok(mut keyframes) if keyframes.len() >= 2 => {
                keyframes.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
                for kf in &mut keyframes {
                    kf.x = kf.x.clamp(0.05, 0.95);
                }
                build_dynamic_crop(&keyframes)
            }
            _ => build_static_crop(center_x_norm),
        }
    } else {
        build_static_crop(center_x_norm)
    };

    // Build the final filter chain
    let mut filter_chain = String::new();
    
    let wm_config: Option<serde_json::Value> = watermark_json_str
        .as_ref()
        .and_then(|s| serde_json::from_str(s).ok());

    if let Some(ref config) = wm_config {
        let wm_type = config["type"].as_str().unwrap_or("image");
        let pos = config["pos"].as_str().unwrap_or("top_left");
        let opacity = config["opacity"].as_f64().unwrap_or(1.0);
        
        let (pos_x, pos_y) = match pos {
            "top_left" => ("20", "20"),
            "top_right" => ("W-w-20", "20"),
            "bottom_left" => ("20", "H-h-20"),
            "bottom_right" => ("W-w-20", "H-h-20"),
            "center" => ("(W-w)/2", "(H-h)/2"),
            _ => ("20", "20") // Default
        };

        if wm_type == "image" {
            let path = config["path"].as_str().unwrap_or("")
                .replace("\\", "\\\\")
                .replace("'", "\\\\'");
            // Use format expression for opacity via colorchannelmixer
            filter_chain.push_str(&format!(
                "movie='{}',colorchannelmixer=aa={:.2}[wm];[in]{}[cropped];[cropped][wm]overlay={}:{}[wm_out]", 
                path, opacity, crop_filter, pos_x, pos_y
            ));
        } else if wm_type == "text" {
            // Text watermark uses drawtext instead of complex layout chaining
            let text = config["text"].as_str().unwrap_or("")
                .replace("\\", "\\\\")
                .replace(":", "\\:")
                .replace("'", "\\'");
            let drawtext = format!(
                "drawtext=text='{}':fontcolor=white@{:.2}:fontsize=40:fontfile='DejaVuSans-Bold.ttf':x={}:y={}",
                text, 
                opacity, 
                pos_x.replace("W", "TEMP_W").replace("w", "tw").replace("TEMP_W", "w"), 
                pos_y.replace("H", "TEMP_H").replace("h", "th").replace("TEMP_H", "h")
            );
            // x/y vars in drawtext are w/h (video width/height) and tw/th (text width/height)
            let drawtext_fixed = drawtext
                .replace("tw-tW", "w-tw") // Correct upper/lower W variables back (hacky cleanup of overlay variables)
                .replace("th-tH", "h-th");
            // Since drawtext doesn't merge inputs, it just chains directly after crop
            filter_chain.push_str(&format!("{},{}[wm_out]", crop_filter, drawtext_fixed));
        } else {
            filter_chain.push_str(&format!("{}[wm_out]", crop_filter));
        }
    } else {
        filter_chain.push_str(&crop_filter);
    }
    
    if let Some(sub_path) = subtitle_path {
        if !sub_path.is_empty() {
            let escaped_path = sub_path
                .replace("\\", "\\\\")
                .replace(":", "\\:")
                .replace("'", "\\'");
                
            let sub_filter = if sub_path.ends_with(".ass") {
                format!("ass='{}'", escaped_path)
            } else {
                format!(
                    "subtitles='{}':force_style='Fontname=Sans,Fontsize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=1,Shadow=1,Alignment=2'",
                    escaped_path 
                )
            };
            
            if wm_config.is_some() {
                // The prev_chain already ends with [wm_out], so just use it as input
                let prev_chain = filter_chain.clone();
                filter_chain = format!("{};[wm_out]{}", prev_chain, sub_filter);
            } else {
                // Simple comma-separated list
                filter_chain.push_str(",");
                filter_chain.push_str(&sub_filter);
            }
        }
    }

    let output = Command::new("ffmpeg")
        .args(&[
            "-nostdin",
            "-y",
            "-ss", &start.to_string(),
            "-i", &input_path,
            "-t", &duration.to_string(),
            "-vf", &filter_chain,
            "-c:v", "libx264",
            "-preset", "fast",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            &output_path,
        ])
        .output()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to run ffmpeg: {}", e)))?;

    if !output.status.success() {
        return Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(
            String::from_utf8_lossy(&output.stderr).to_string(),
        ));
    }

    Ok(output_path)
}

#[pymodule]
fn clipperr_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_version, m)?)?;
    m.add_function(wrap_pyfunction!(extract_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(render_clip, m)?)?;
    Ok(())
}
