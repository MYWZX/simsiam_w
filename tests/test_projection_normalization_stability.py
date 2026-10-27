"""FP16 zero outputs must not create NaN in shared projections or SimSiam."""
import unittest
import torch
from models.two_stage_models import simsiam_pair_loss
from test_dual_quality_residual import residual_config
from test_projection_ablations import make

class ProjectionNormalizationStability(unittest.TestCase):
    def test_half_zero_projection_outputs_and_simsiam_are_finite(self):
        c=residual_config();c['model']['shared_simsiam_projection']=True
        m=make(c).half()
        with torch.no_grad():
            for module in [m.iqa.shared_image_projector,m.iqa.shared_text_projector]:
                for p in module.parameters():p.zero_()
        feature=torch.zeros(3,8,dtype=torch.float16,requires_grad=True)
        image=m._project_image(feature);text=m._project_text(feature)
        for out in [image,text]:
            for value in out.values():self.assertTrue(torch.isfinite(value).all())
        loss=simsiam_pair_loss(image['p'],text['z_proj'],text['p'],image['z_proj'])
        self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertTrue(torch.isfinite(feature.grad).all())
        p=torch.zeros(3,8,dtype=torch.float16,requires_grad=True)
        loss=simsiam_pair_loss(p,p.detach(),p,p.detach())
        self.assertTrue(torch.isfinite(loss));loss.backward();self.assertTrue(torch.isfinite(p.grad).all())

    def test_regular_half_vectors_have_identical_normalized_values(self):
        c=residual_config();c['model']['shared_simsiam_projection']=True
        m=make(c,joint=False).half();x=torch.randn(3,8,dtype=torch.float16)
        with torch.no_grad():
            raw=m.shared_image_projector(x)
            expected=torch.nn.functional.normalize(raw,dim=-1)
            actual=m.project_shared_image(x)
        self.assertTrue(torch.isfinite(expected).all())
        torch.testing.assert_close(expected,actual,rtol=0,atol=0)

if __name__=='__main__':unittest.main()
