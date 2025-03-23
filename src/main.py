import subprocess

def run_predict_svm(args):
    """Run the SVM prediction script."""
    print("Making prediction with SVM model...")
    
    cmd = [
        'python', 'src/predict_svm.py',
        '--model-path', 'src/models/saved'  # Fixed path to model directory instead of specific model file
    ]
    
    if args.interactive:
        cmd.append('--interactive')
    else:
        if args.cpu is not None:
            cmd.extend(['--cpu', str(args.cpu)])
        if args.memory is not None:
            cmd.extend(['--memory', str(args.memory)])
        if args.network is not None:
            cmd.extend(['--network', str(args.network)])
        if args.disk is not None:
            cmd.extend(['--disk', str(args.disk)])
        if args.error is not None:
            cmd.extend(['--error', str(args.error)])
        if args.response is not None:
            cmd.extend(['--response', str(args.response)])
        if args.restarts is not None:
            cmd.extend(['--restarts', str(args.restarts)])
        if args.throttle is not None:
            cmd.extend(['--throttle', str(args.throttle)])
    
    if args.output_path:
        cmd.extend(['--output-path', args.output_path])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error making prediction:")
        print(result.stderr)
        return False
    
    print(result.stdout)
    return True 