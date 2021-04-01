LIMB_SEQ = JSON.parse("[[5,7], [7, 9], [5, 6], [6, 8], [8, 10], [5, 11], [11, 12], [6, 12], [11, 13], [13, 15], [12, 14], [14, 16]]")

MISSING_VALUE = -1

function drawLine(ctx, from, to) {
    ctx.strokeStyle = 'white'
    ctx.beginPath();
    ctx.moveTo(from[1], from[0]);
    ctx.lineTo(to[1], to[0]);
    ctx.stroke();
}
// function for drawing the poses
function drawPose(canvas, pose){
    if(canvas.getContext){
        canvas.width = 176
        canvas.height = 256
        var ctx = canvas.getContext('2d');
        
        LIMB_SEQ.forEach(limb => {
            if(pose[limb[0]][0] === MISSING_VALUE | pose[limb[0]][1] === MISSING_VALUE | pose[limb[1]][0] === MISSING_VALUE | pose[limb[1]][1] === MISSING_VALUE) {
                return;
            } else {
                drawLine(ctx, pose[limb[1]], pose[limb[0]]);
            }
        })
        ctx.fillStyle = 'red';
        pose.forEach(limb => {
            if (limb[0] === MISSING_VALUE | limb[1] === MISSING_VALUE) {
                return;
            }
            ctx.fillRect(limb[1], limb[0], 5, 5);
        });
    }
}
